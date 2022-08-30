import os
import cv2 as cv
import glob

# img_path = "E:/xyw/R2D2/气管镜图像结果/融合图_0.8distance"
#
# video_path = "E:/xyw/R2D2/气管镜图像结果"
#
# if not os.path.exists(video_path):
#     os.makedirs(video_path)
#
# fps = 20
#
# frames = sorted(os.listdir(img_path))
#
# # img = cv.imread(os.path.join(img_path,"/"+frames[0]))
# img_size = (640,480)
# seq_name = os.path.dirname(img_path).split('/')[-1]
#
# video_path = os.path.join(video_path,seq_name + '.avi')
# fourcc = cv.VideoWriter_fourcc(*'XVID')
#
# videowriter = cv.VideoWriter("fusion.avi",fourcc,fps,img_size)
#
# for frame in frames:
#     f_path = os.path.join(img_path,frame)
#     image = cv.imread(f_path)
#     videowriter.write(image)
#     print(frame+" has been written")
#
# videowriter.release()

def imgs2video(imgs_dir,save_name):
    fps=20
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    videowriter = cv.VideoWriter(save_name,fourcc,fps,(480,640))

    imgs = glob.glob(os.path.join(imgs_dir,'*.jpg'))
    imgs=sorted(imgs)

    for i in range(len(imgs)):
        imgname = os.path.join(imgs_dir,'core-{:02d}.jpg'.format(i))
        imgname = os.path.join(imgs_dir,imgs[i])
        frame = cv.imread(imgname)
        videowriter.write(frame)
        print(i)
    videowriter.release()

if __name__ == '__main__':
    img_path = "E:/xyw/R2D2/气管镜图像结果/融合图_0.8distance/"

    video_path = "E:/xyw/R2D2/气管镜图像结果/fusion.avi"
    imgs2video(img_path,video_path)