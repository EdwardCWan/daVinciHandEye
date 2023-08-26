"""
This script is to calibrate a single (mono) camera.
It ends up with the intrinsic, extrinsic matrix and distortion params of the camera.
"""
import glob
import cv2 as cv
import numpy as np
import os
import shutil

def monoCamCali(imgs_path_ls: list, num_frame_used: int, corners_x: int, corners_y: int, square_size: float, verbose2save: bool, verbose2show: bool):
    # termination criteria:
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare the object points in 3D w.r.t the {board} frame:
    objp = np.zeros((corners_x * corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    objp[:, 0:2] = objp[:, 0:2] * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # detect the chess board corners:
    for fname in imgs_path_ls[0:num_frame_used]:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (corners_x, corners_y), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        if verbose2show:
            cv.drawChessboardCorners(img, (corners_x, corners_y), corners2, ret)
            cv.imshow('img', img)
            # cv.waitKey(1000)
    cv.destroyAllWindows()

    # camera calibration:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # check the reprojection error:
    mean_error = 0
    for i in range(len(objpoints)):
        # we transform the objectpoint to an imgaepoint (pred):
        imgpoints_pred, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # the second norm between detected imagepoint and predicted image point are the error:
        error = cv.norm(imgpoints[i], imgpoints_pred, cv.NORM_L2)/len(imgpoints_pred)
        print(f"{imgs_path_ls[i]}\t -- {error:.4f}")
        mean_error += error
    print("total error: {}".format(mean_error/len(objpoints)) )

    # save the camera model:
    if verbose2save:
        save_dir = f"monoCamCali/cameraModel/{imgs_path_ls[0].split('/')[-2]}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        else:
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)

        np.save(f"{save_dir}/intrinsic.npy", mtx)
        np.save(f"{save_dir}/distortion.npy", dist)
        for i in range(num_frame_used):
            np.save(f"{save_dir}/rotation{i+1:02}.npy", rvecs[i])
            np.save(f"{save_dir}/translation{i+1:02}.npy", tvecs[i])
    
    return mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    # 1. get all the dir of the images:
    camera_type = "right"
    imgs_dir = f"monoCamCali/images/{camera_type}"
    imgs_path = glob.glob(f"{imgs_dir}/*.png")
    imgs_path.sort()

    # 2. all image path will be stored in a list, pass to the calibration function
    mtx, dist, rvecs, tvecs = monoCamCali(
        imgs_path_ls=imgs_path,
        num_frame_used=13,
        corners_x=13,
        corners_y=10,
        square_size=0.3, # cm
        verbose2save=False,
        verbose2show=False
        )