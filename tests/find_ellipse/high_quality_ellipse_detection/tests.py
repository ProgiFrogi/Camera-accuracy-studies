import cv2
ellipses = []
with open('utils/find_ellipse/high_quality_ellipse_detection/output_candidate_ellipses.txt','r') as f:
    text = f.read().split('\n')
    for i in text:
        if i == '':
            continue
        ellipses.append(list(map(float,i.split(' '))))

img = cv2.imread('tmp/hehe4.png')

for i in ellipses:
    img = cv2.ellipse(img,list(map(int,i[0:2])),list(map(int,i[2:4])),i[4],0,360,(255,255,255),3)
cv2.imshow("heh",img)
cv2.waitKey(0)
cv2.destroyAllWindows()