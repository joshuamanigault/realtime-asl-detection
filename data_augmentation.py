import os
import cv2
import numpy as np

DATA_DIR = './dataV2'


def augment_image(img):
    augmented_images = [
        img,
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.flip(img, 1),  # Horizontal flip
        cv2.flip(img, 0)  # Vertical flip
    ]

    # Color Jittering
    augmented_images.extend(color_jittering(img))

    return augmented_images


def color_jittering(img):
    jittered_images = []

    for _ in range(5):
        # Convert to HSV (Hue, Saturation, Value) to adjust brightness and color
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Randomly change the brightness
        value_change = np.random.randint(-50, 50)
        hsv_img[:, :, 2] = cv2.add(hsv_img[:, :, 2], value_change)

        # Randomly change the saturation
        saturation_change = np.random.uniform(0.5, 1.5)
        hsv_img[:, :, 1] = cv2.multiply(hsv_img[:, :, 1], saturation_change)

        # Convert back to BGR
        jittered_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        jittered_images.append(jittered_img)

    return jittered_images


for class_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, class_name)
    print(f'Augmenting data for class {class_name}')

    counter = len(os.listdir(class_dir))  # Start counter from the existing number of images
    for filename in os.listdir(class_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)

            augmented_images = augment_image(img)
            for aug_img in augmented_images:
                aug_img_path = os.path.join(class_dir, f'{counter}.jpg')
                cv2.imwrite(aug_img_path, aug_img)
                counter += 1

print('Data augmentation complete.')
