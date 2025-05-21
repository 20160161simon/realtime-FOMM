import cv2

def cropping_frame(frame, target_size=(256, 256)):
    # frame.shape = (480, 640, 3)
    frame = cv2.resize(frame, target_size)
    frame = cv2.flip(frame, 1)  # flip horizontally
    return frame

def combine_frames(camera_input, FOMM_output):

    # add border
    border_size = 6
    camera_input = cv2.copyMakeBorder(
        camera_input, border_size, border_size, border_size, border_size // 2,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    FOMM_output = cv2.copyMakeBorder(
        FOMM_output, border_size, border_size, border_size, border_size - border_size // 2,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # add label
    label_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    thickness = 1
    text_left = "Camera Input"
    text_right = "FOMM Output"

    camera_labeled = cv2.copyMakeBorder(
        camera_input, 0, label_height, 0, 0,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    FOMM_labeled = cv2.copyMakeBorder(
        FOMM_output, 0, label_height, 0, 0,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    (text_w, text_h), _ = cv2.getTextSize(text_left, font, font_scale, thickness)
    x_left = (camera_labeled.shape[1] - text_w) // 2
    cv2.putText(
        camera_labeled, text_left, (x_left, 256 + 2 + label_height - 10),
        font, font_scale, font_color, thickness, cv2.LINE_AA)

    (text_w, text_h), _ = cv2.getTextSize(text_right, font, font_scale, thickness)
    x_right = (FOMM_labeled.shape[1] - text_w) // 2
    cv2.putText(
        FOMM_labeled, text_right, (x_right, 256 + 2 + label_height - 10),
        font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    combined = cv2.hconcat([camera_labeled, FOMM_labeled])
    return combined
