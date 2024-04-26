import cv2 as cv

DIR_PATH = "src/mocap/synch_saw_22Apr/"
FILETYPE = '.mp4'
FOURCC = cv.VideoWriter_fourcc(*'mp4v')

name_chair = DIR_PATH + "Saw_Synch_01" + FILETYPE
name_monit = DIR_PATH + "Saw_Synch_02" + FILETYPE
cap_monit = cv.VideoCapture(2)
cap_chair = cv.VideoCapture(1)
frame_width = int(cap_monit.get(3))
frame_height = int(cap_chair.get(4))


out_chair = cv.VideoWriter(name_chair, FOURCC, 20, (frame_width, frame_height))
out_monit = cv.VideoWriter(name_monit, FOURCC, 20, (frame_width, frame_height))

# i = 0
# while True:
#     r_c, f_c = cap_chair.read()
#     r_m, f_m = cap_monit.read()

#     if r_c and r_m:
#         f_c = cv.flip(f_c, 1)
#         f_m = cv.flip(f_m, 1)
#         cv.imshow('frame_chair', f_c)
#         cv.imshow('frame_monitor', f_m)

#         k = cv.waitKey(1)
#         if k%256 == 27:
#             # ESC pressed
#             print("Escape hit, closing...")
#             break
#         elif k%256 == 32:
#             # SPACE pressed
#             cv.imwrite(DIR_PATH + f'calib/cam01_{i}.png', f_c)
#             cv.imwrite(DIR_PATH + f'calib/cam02_{i}.png', f_m)
#             print(f'Wrote image {i}')
#             i += 1

frames_ctr = 0
while(cap_chair.isOpened() and cap_monit.isOpened()):
    ret_c, frame_c = cap_chair.read()
    ret_m, frame_m = cap_monit.read()

    if ret_c==True and ret_m == True:
        frame_c, frame_m = cv.flip(frame_c,1), cv.flip(frame_m,1)
        frames_ctr += 1
        cv.imshow('frame_chair',frame_c)
        cv.imshow('frame_monitor',frame_m)
        out_chair.write(frame_c)
        out_monit.write(frame_m)

        if (cv.waitKey(1) & 0xFF == ord('q')):
            break

    else:
        break

cap_chair.release()
cap_monit.release()
out_monit.release()
out_chair.release()
cv.destroyAllWindows()