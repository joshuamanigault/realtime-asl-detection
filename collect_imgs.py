import os
import cv2

DATA_DIR = './dataV2'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_name = input(f'Enter the name for class {j}: ')
    class_dir = os.path.join(DATA_DIR, class_name)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_name} (ID: {j})')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == 27:  # 27 is the ASCII code for the ESC key
            cap.release()
            cv2.destroyAllWindows()
            print("Script terminated by user.")
            exit()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            print("Script terminated by user.")
            exit()
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
