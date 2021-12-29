# import packages
import numpy as np
import cv2
import glob

def rgb_select(img, r_thresh, g_thresh, b_thresh):
    r_channel = img[:,:,0]
    g_channel=img[:,:,1]
    b_channel = img[:,:,2]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    g_binary = np.zeros_like(g_channel)
    g_binary[(r_channel > g_thresh[0]) & (r_channel <= g_thresh[1])] = 1

    b_binary = np.zeros_like(b_channel)
    b_binary[(r_channel > b_thresh[0]) & (r_channel <= b_thresh[1])] = 1

    combined = np.zeros_like(r_channel)
    combined[((r_binary == 1) & (g_binary == 1) & (b_binary == 1))] = 1
    return combined


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def color_thresh(image, s_thresh, l_thresh, b_thresh, v_thresh):
    luv= cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    s_channel = hsv[:,:,1]
    b_channel=lab[:,:,2]
    l_channel = luv[:,:,0]
    v_channel= hsv[:,:,2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    b_binary = np.zeros_like(b_channel)
    b_binary[(s_channel > b_thresh[0]) & (s_channel <= b_thresh[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(s_channel > l_thresh[0]) & (s_channel <= l_thresh[1])] = 1

    v_binary = np.zeros_like(v_channel)
    v_binary[(s_channel > v_thresh[0]) & (s_channel <= v_thresh[1])] = 1

    combined = np.zeros_like(s_channel)
    combined[((s_binary == 1) & (b_binary == 1) & (l_binary == 1) & (v_binary == 1))] = 1

    return combined


def color_gradient_threshold(image_undistorted):
    ksize = 15
    luv= cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LUV)
    hls = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image_undistorted,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2LAB)
    s_channel = hsv[:,:,1]

    gradx = abs_sobel_thresh(image_undistorted,orient='x',sobel_kernel=ksize,thresh=(50,90))
    grady = abs_sobel_thresh(image_undistorted,orient='y',sobel_kernel=ksize,thresh=(30,90))
    c_binary = color_thresh(image_undistorted,s_thresh=(70,100),l_thresh=(60,255),b_thresh=(50,255),v_thresh=(150,255))
    rgb_binary = rgb_select(image_undistorted,r_thresh=(225,255),g_thresh=(225,255),b_thresh=(0,255))
    combined_binary = np.zeros_like(s_channel)

    combined_binary[((gradx == 1) & (grady == 1) | (c_binary == 1) | (rgb_binary==1))] = 255
    color_binary = combined_binary
    return color_binary, combined_binary


# 透视变换
def perspective_transform(image_undistorted, combined_binary):
    top_left = [560, 470]
    top_right = [730, 470]
    bottom_right = [1080, 720]
    bottom_left = [200, 720]

    top_left_dst = [200,0]
    top_right_dst = [1100,0]
    bottom_right_dst = [1100,720]
    bottom_left_dst = [200,720]
    img_size = (image_undistorted.shape[1], image_undistorted.shape[0])
    src = np.float32([top_left,top_right, bottom_right, bottom_left] )
    dst = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped  = cv2.warpPerspective(combined_binary, M, img_size)
    return warped, Minv
	

def finding_line(warped):
    # 将warped中从360行开始加到720行；
    histogram2 = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((warped, warped, warped))*255
    midpoint = int(histogram2.shape[0]/2)

    leftx_base = np.argmax(histogram2[:midpoint])
    rightx_base = np.argmax(histogram2[midpoint:])+midpoint
    nwindows = 5
    window_height = int(warped.shape[0]/nwindows)
    nonzero = warped.nonzero()
    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = warped.shape[0]-(window+1)*window_height
        win_y_high = warped.shape[0]-window*window_height
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
  
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    
    # 找出左车道线附近的像素点序号；
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    # 找出右车道线附近的像素点序号；
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    return left_fitx, right_fitx, out_img, left_fit, right_fit, left_lane_inds, right_lane_inds


def sliding_window(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    # 找出左车道线附近的像素点序号；
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
     # 找出右车道线附近的像素点序号；
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # 找到车道线像素点的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # 重新拟合出一条二次曲线
    #left_fit = np.polyfit(lefty, leftx, 2)
    #right_fit = np.polyfit(righty, rightx, 2)
    # 产生画图的点
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # 产生一张空图
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # 画图的颜色
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
   
    return left_fitx, right_fitx, left_line_pts,right_line_pts,window_img, out_img,left_lane_inds,right_lane_inds


def CalculateCurvature(binary_image, left_fit, right_fit, l_lane_inds, r_lane_inds):

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    img_size = (binary_image.shape[1], binary_image.shape[0])
   
    ploty = np.linspace(0, img_size[1]-1, img_size[1])
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/720  
    xm_per_pix = 3.7/960     # meters per pixel in y dimension 
    
    # 找到图像中不为零的所有像素点的像素坐标
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 将这些不为零的像素点坐标分成x，y车道线中
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    # 将这些像素点对应到世界坐标系中，然后拟合成二次曲线
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # 计算曲线的曲率
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # 左右车道线曲率平均
    avg_curverad = (left_curverad + right_curverad) / 2
    
    ## 以下计算本车在车道线中心的位置

    dist_from_center = 0.0
    # assume the camera is centered in the vehicle
    ###camera_pos = img_size[1] / 2
    if right_fit is not None:
        if left_fit is not None:
            # 摄像头位于图像中间，也是本车的中心
            camera_pos = img_size[0] / 2
            ###RESUBMIT - END
            
            # 左右车道线最底端x坐标
            left_lane_pix = np.polyval(left_fit, binary_image.shape[0])
            right_lane_pix = np.polyval(right_fit, binary_image.shape[0])
            # 左右车道线中点x坐标
            center_of_lane_pix = (left_lane_pix + right_lane_pix) / 2
            # 摄像头（本车中心）与车道线中心的距离
            dist_from_center = (camera_pos - center_of_lane_pix) * 3.7/960

    return  avg_curverad, dist_from_center


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def apply_region_of_interest_mask(image):
    x_factor = 40
    y_factor = 60
    vertices = np.array([[
        (0,image.shape[0]),
        (((image.shape[1]/2)- x_factor), (image.shape[0]/2)+ y_factor), 
         (((image.shape[1]/2) + x_factor), (image.shape[0]/2)+ y_factor), 
         (image.shape[1],image.shape[0])]], dtype=np.int32)

    return region_of_interest(image, vertices)


def overlay_text_on_image (image, avg_curverad, dist_from_center):
    new_img = np.copy(image)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255,255,255)
    
    num_format = '{:04.2f}'
    text = 'Radius of Curvature: ' + num_format.format(avg_curverad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, font_color, 2, cv2.LINE_AA)
    
    direction = 'left'
    if dist_from_center > 0:
        direction = 'right'
    abs_dist = abs(dist_from_center)
    text = 'Vehicle is ' + num_format.format(abs_dist) + ' m ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, font_color, 2, cv2.LINE_AA)
    
    return new_img


def main_pipline(image_0):
    
    #1 畸变矫正
    # image_undistorted = cv2.undistort(image_0, mtx, dist, None, mtx)

    #2 颜色与梯度阈
    color_binary, combined_binary = color_gradient_threshold(image_0)

    #3 感兴趣区域
    masked = apply_region_of_interest_mask(combined_binary)

    #4 透视变换
    warped_0, Minv = perspective_transform(masked, combined_binary)
    
    #5 滑动窗车道线提取
    left_fitx, right_fitx, out_img,left_fit, right_fit,left_lane_inds,right_lane_inds = finding_line(warped_0)
    
    #6 计算车道线的曲率
    avg_curverad, dist_from_center = CalculateCurvature(warped_0,left_fit, right_fit, left_lane_inds, right_lane_inds)
    
    #7 在图像上画车道线
    ploty = np.linspace(0, warped_0.shape[0]-1, warped_0.shape[0])
    warp_zero = np.zeros_like(warped_0).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_0.shape[1], image_0.shape[0])) 
    result = cv2.addWeighted(image_0, 1, newwarp, 0.3, 0)

    #8 图像上显示文字
    result = overlay_text_on_image (result, avg_curverad, dist_from_center)
    return result

def process_image(image):
    result = main_pipline(image)
    return result


cap = cv2.VideoCapture('project_video.mp4')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ret_frame = process_image(frame)
    ret_frame = cv2.cvtColor(ret_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame',ret_frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Break the loop
  else: 
    break

cap.release()
cv2.destroyAllWindows()
