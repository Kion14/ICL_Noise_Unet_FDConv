import numpy as np
import cv2
import scipy.ndimage
import random

def elastic_transform(image, mask, alpha, sigma):
    random_state = np.random.RandomState(42)
    shape = image.shape[:2]
    
    dx = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distorted_image = scipy.ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)
    distorted_mask = scipy.ndimage.map_coordinates(mask, indices, order=1, mode='reflect').reshape(mask.shape)
    
    return distorted_image, distorted_mask

def zoom_image(image, mask, zoom_factor=1.2):
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    
    y1, x1 = (h - new_h) // 2, (w - new_w) // 2
    y2, x2 = y1 + new_h, x1 + new_w
    
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    resized_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return resized_image, resized_mask

def add_gaussian_noise(image, mean=0, std=0.01):
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 1)  # Keep pixel values between 0-1

def adjust_brightness_contrast(image, alpha=1.2, beta=0.2):
    return np.clip(alpha * image + beta, 0, 1)

################################################################### NIUEW
# def random_intensity_augmentation(image, p=0.8):
#     """
#     Brightness + contrast + gamma augmentation.
#     image expected in [0, 1], shape [H, W, C]
#     """
#     if random.random() > p:
#         return image

#     image = image.astype(np.float32)

#     # brightness/contrast
#     alpha = random.uniform(0.75, 1.35)  # contrast
#     beta = random.uniform(-0.15, 0.15)  # brightness
#     image = alpha * image + beta
#     image = np.clip(image, 0, 1)

#     # gamma
#     gamma = random.uniform(0.7, 1.5)
#     image = np.power(image, gamma)

#     return np.clip(image, 0, 1).astype(np.float32)



def random_intensity_augmentation(image, p=0.8):
    if random.random() > p:
        return image

    image = image.astype(np.float32)

    # contrast
    alpha = random.uniform(0.75, 1.35)

    # brightness
    beta = random.uniform(-0.15, 0.15)

    image = alpha * image + beta
    image = np.clip(image, 0, 1)

    # gamma
    gamma = random.uniform(0.7, 1.5)
    image = np.power(image, gamma)

    return np.clip(image, 0, 1).astype(np.float32)


def random_invert_intensity(image, p=0.3):
    """
    Random intensity inversion:
    dark nuclei <-> bright nuclei
    """
    if random.random() > p:
        return image

    image = image.astype(np.float32)

    return (1.0 - image).astype(np.float32)

def random_he_augmentation(image, p=0.5):
    if random.random() > p:
        return image

    image = image.astype(np.float32)

    # brightness / contrast
    alpha = random.uniform(0.85, 1.20)
    beta = random.uniform(-0.08, 0.08)
    image = np.clip(alpha * image + beta, 0, 1)

    # gamma
    gamma = random.uniform(0.85, 1.25)
    image = np.power(image, gamma)

    # hue / saturation
    if random.random() < 0.5:
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-5, 5)) % 180
        hsv[:, :, 1] *= random.uniform(0.85, 1.20)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # lichte blur
    if random.random() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return np.clip(image, 0, 1).astype(np.float32)





# def random_color_augmentation(image, p=0.5):
#     """
#     Color/stain augmentation:
#     - saturation jitter
#     - hue shift
#     - channel dropout
#     - channel shuffle

#     image expected in [0, 1], shape [H, W, 3]
#     """
#     if random.random() > p:
#         return image

#     image = image.astype(np.float32)

#     # saturation + hue via HSV
#     img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
#     hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

#     # saturation jitter
#     sat_factor = random.uniform(0.6, 1.5)
#     hsv[:, :, 1] *= sat_factor

#     # hue shift
#     hue_shift = random.uniform(-10, 10)
#     hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

#     image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

#     # channel dropout
#     if random.random() < 0.25:
#         ch = random.randint(0, 2)
#         image[:, :, ch] = 0.0

#     # channel shuffle
#     if random.random() < 0.25:
#         order = np.random.permutation(3)
#         image = image[:, :, order]

#     return np.clip(image, 0, 1).astype(np.float32)

# def random_grayscale(image, p=0.3):
#     """
#     Random grayscale augmentation.
#     Converts RGB -> grayscale -> back to RGB.
#     """
#     if random.random() > p:
#         return image

#     image = image.astype(np.float32)

#     gray = cv2.cvtColor(
#         (image * 255).astype(np.uint8),
#         cv2.COLOR_RGB2GRAY
#     )

#     gray = gray.astype(np.float32) / 255.0

#     # back to 3 channels
#     gray_rgb = np.stack([gray, gray, gray], axis=-1)

#     return np.clip(gray_rgb, 0, 1).astype(np.float32)
###################################################################



def augment_data(image_data, context = False, target_size=(512, 512), max_rotations=50, max_zoom=1.4, max_elastic=True):
    augmented_data = []
    if context is False:
        for image, mask in image_data:
            augmented_data.append((image, mask))
            
            # Horizontal Flip
            augmented_data.append((np.fliplr(image), np.fliplr(mask)))
            
            # Vertical Flip
            augmented_data.append((np.flipud(image), np.flipud(mask)))
            
            # Limit number of rotations to reduce memory usage
            rotations = [90, 180, 270]  # Only applying 90 and 270 degrees to limit memory usage
            for angle in rotations[:max_rotations]:  # Limit number of rotations applied
                rotated_image = np.rot90(image, k=angle // 90)
                rotated_mask = np.rot90(mask, k=angle // 90)

                rotated_image = cv2.resize(rotated_image, target_size)  # Resize to (height, width)
                rotated_mask = cv2.resize(rotated_mask, target_size)
                rotated_image = cv2.normalize(cv2.resize(rotated_image, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
                rotated_mask = cv2.normalize(cv2.resize(rotated_mask, target_size, interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            

                augmented_data.append((rotated_image, rotated_mask))
            
            # Zooming with limited zoom factor to reduce memory usage
            if max_zoom:
                zoomed_image, zoomed_mask = zoom_image(image, mask, zoom_factor=max_zoom)  # Use smaller zoom factor
                zoomed_image = cv2.normalize(cv2.resize(zoomed_image, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
                zoomed_mask = cv2.normalize(cv2.resize(zoomed_mask, target_size, interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            
                augmented_data.append((zoomed_image, zoomed_mask))
            
            # Elastic Transformations with optional reduced parameters for memory efficiency
            if max_elastic:
                elastic_image, elastic_mask = elastic_transform(image, mask, alpha=50, sigma=10)  # Lighter elastic transform
                elastic_image = cv2.normalize(cv2.resize(elastic_image, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
                elastic_mask = cv2.normalize(cv2.resize(elastic_mask, target_size, interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0
    
                augmented_data.append((elastic_image, elastic_mask))
            
            # Gaussian Noise
            noisy_image = add_gaussian_noise(image)
            noisy_image = cv2.normalize(cv2.resize(noisy_image, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
          
            augmented_data.append((noisy_image, mask))
            
            # Brightness and Contrast Adjustment
            bright_contrast_image = adjust_brightness_contrast(image)
            bright_contrast_image = cv2.normalize(cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
  
            augmented_data.append((bright_contrast_image, mask))

        return augmented_data
    else:
        for image1, mask1, image2, mask2 in image_data:
            augmented_data.append((image1, mask1, image2, mask2))
            
            # Horizontal Flip
            augmented_data.append((np.fliplr(image1), np.fliplr(mask1), np.fliplr(image2), np.fliplr(mask2)))
            
            # Vertical Flip
            augmented_data.append((np.flipud(image1), np.flipud(mask1), np.flipud(image2), np.flipud(mask2)))
            
            # Limit number of rotations to reduce memory usage
            rotations = [90, 180, 270]  # Only applying 90 and 270 degrees to limit memory usage
            for angle in rotations[:max_rotations]:  # Limit number of rotations applied
                rotated_image1 = np.rot90(image1, k=angle // 90)
                rotated_mask1 = np.rot90(mask1, k=angle // 90)
                rotated_image2 = np.rot90(image2, k=angle // 90)
                rotated_mask2 = np.rot90(mask2, k=angle // 90)

                rotated_image1 = cv2.resize(rotated_image1, target_size)  # Resize to (height, width)
                rotated_mask1 = cv2.resize(rotated_mask1, target_size)
                rotated_image2 = cv2.resize(rotated_image2, target_size)  # Resize to (height, width)
                rotated_mask2 = cv2.resize(rotated_mask2, target_size)

                augmented_data.append((rotated_image1, rotated_mask1, rotated_image2, rotated_mask2))


        return augmented_data
            

def upsample_labels(*data_sets):
    # Vind de maximale grootte van alle datasets
    max_size = max(len(data_set) for data_set in data_sets)
    print(f"Upsampling to {max_size} samples per class", flush=True)

    # Upsample alle datasets naar de maximale grootte
    upsampled_data_sets = []
    for data_set in data_sets:
        # Als de dataset al de maximale grootte heeft, sla over
        if len(data_set) == max_size:
            upsampled_data_sets.append(data_set)
            continue

        # Herhaal willekeurig geselecteerde samples om de dataset te vergroten
        upsampled_data_set = data_set[:]
        while len(upsampled_data_set) < max_size:
            extra_samples = random.choices(data_set, k=max_size - len(upsampled_data_set))
            upsampled_data_set.extend(extra_samples)

        upsampled_data_sets.append(upsampled_data_set)

    return upsampled_data_sets