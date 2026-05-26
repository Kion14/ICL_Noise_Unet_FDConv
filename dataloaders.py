import numpy as np
import cv2
import os
import random
import os
import pydicom
import nibabel as nib
from glob import glob
import csv
import pandas as pd
import json
from PIL import Image
import random
from collections import Counter



def read_and_split_busbra_data(
    busbra_root="./BUSBRA",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    target_size=(192, 192)
):
    """
    Reads BUSBRA dataset and splits into train/val/test with similar class proportions.
    Returns: X, V, Y lists of (image, mask, class) tuples.
    """

    random.seed(42)
    classes = ['benign', 'malignant']
    all_data = {cls: [] for cls in classes}

    # Read class info from bus_data.csv
    csv_path = os.path.join(busbra_root, "bus_data.csv")
    df = pd.read_csv(csv_path)
    # Expect columns: 'filename', 'class' (adjust if needed)

    # Build mapping from image filename to class
    filename_to_class = dict(zip(df['ID'], df['Pathology']))
    # Find all images and masks
    img_dir = os.path.join(busbra_root, "Images")
    mask_dir = os.path.join(busbra_root, "Masks")
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    for img_file in img_files:
        base = img_file[4:-4]  # remove .png
        mask_file = f"mask_{base}.png"
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        # Get class from CSV
        cls = filename_to_class.get(img_file[:-4], None)
        if cls not in classes:
            continue  # skip if class not found or not in target classes
        if not os.path.exists(mask_path):
            continue  # skip if mask missing
        all_data[cls].append((img_path, mask_path, cls))
    # Split each class separately
    X, V, Y = [], [], []
    for cls in classes:
        items = all_data[cls]
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        X.extend(items[:n_train])
        V.extend(items[n_train:n_train+n_val])
        Y.extend(items[n_train+n_val:])

    # Optionally shuffle the combined sets
    random.shuffle(X)
    random.shuffle(V)
    random.shuffle(Y)
    #print(len(X),len(V),len(Y))
    # Read and resize images/masks
    def read_and_resize(pairs):
        data = []
        for img_path, mask_path, cls in pairs:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            img = cv2.normalize(cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            mask = cv2.normalize(cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            #data.append((img, mask, cls))
            data.append((img, mask))

        return data

    X = read_and_resize(X)
    V = read_and_resize(V)
    Y = read_and_resize(Y)
    return X, V, Y

def read_and_split_busi_data(
    busi_root,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    target_size=(192, 192),
    random_seed=42
):
    """
    Reads BUSI dataset and splits into train/val/test with similar class proportions.
    Returns: X, V, Y lists of (image, mask) pairs.
    """
    import glob
    import random

    random.seed(random_seed)
    #classes = ['benign', 'malignant', 'normal']
    classes = ['benign', 'malignant']
    all_data = {cls: [] for cls in classes}

    for cls in classes:
        img_dir = os.path.join(busi_root, cls)
        # Find all images (exclude masks)
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png') and '_mask' not in f]
        for img_file in img_files:
            base = img_file[:-4]  # remove .png
            # Find all masks for this image (some may have multiple, e.g. _mask_1)
            mask_candidates = glob.glob(os.path.join(img_dir, f"{base}_mask*.png"))
            if not mask_candidates:
                continue  # skip if no mask
            # Use the first mask (or you can handle multiple masks as you wish)
            mask_file = mask_candidates[0]
            img_path = os.path.join(img_dir, img_file)
            mask_path = mask_file
            #all_data[cls].append((img_path, mask_path))
            all_data[cls].append((img_path, mask_path, cls))


    # Split each class separately
    X, V, Y = [], [], []
    for cls in classes:
        items = all_data[cls]
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        X.extend(items[:n_train])
        V.extend(items[n_train:n_train+n_val])
        Y.extend(items[n_train+n_val:])

    # Optionally shuffle the combined sets
    random.shuffle(X)
    random.shuffle(V)
    random.shuffle(Y)

    # Read and resize images/masks
    def read_and_resize(pairs):
        data = []
        for img_path, mask_path, cls in pairs:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            img = cv2.normalize(cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            mask = cv2.normalize(cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            data.append((img, mask, cls))
            #data.append((img,mask))
        return data

    X = read_and_resize(X)
    V = read_and_resize(V)
    Y = read_and_resize(Y)
    return X, V, Y

# def split_training_data(image_pairs):
#     length = len(image_pairs)
#     length_train = int(length * 0.70)
#     length_val = int((length * 0.20))

#     X = image_pairs[:length_train]
#     V = image_pairs[length_train:length_train + length_val]
#     Y = image_pairs[length_train + length_val:]

#     return X, V, Y

# def split_leave_one_stain_out(image_pairs, test_stain, val_ratio=0.20, seed=42, print_stats=True):
#     import random
#     from collections import Counter

#     test_set = [item for item in image_pairs if item[2] == test_stain]
#     train_val = [item for item in image_pairs if item[2] != test_stain]

#     random.seed(seed)
#     random.shuffle(train_val)

#     val_len = int(len(train_val) * val_ratio)

#     V = train_val[:val_len]
#     X = train_val[val_len:]
#     Y = test_set

#     if print_stats:
#         print("Leave-one-stain-out split")
#         print("Held-out test stain:", test_stain)
#         print("Train:", len(X), Counter([item[2] for item in X]))
#         print("Val:", len(V), Counter([item[2] for item in V]))
#         print("Test:", len(Y), Counter([item[2] for item in Y]))

#     return X, V, Y


def split_leave_stains_out(image_pairs, test_stains, val_ratio=0.20, seed=42):
    import random
    from collections import Counter

    test_stains = set(test_stains)

    Y = [item for item in image_pairs if item[2] in test_stains]
    train_val = [item for item in image_pairs if item[2] not in test_stains]

    random.seed(seed)
    random.shuffle(train_val)

    val_len = int(len(train_val) * val_ratio)

    V = train_val[:val_len]
    X = train_val[val_len:]

    print("Held-out test stains:", test_stains)
    print("Train:", len(X), Counter([item[2] for item in X]))
    print("Val:", len(V), Counter([item[2] for item in V]))
    print("Test:", len(Y), Counter([item[2] for item in Y]))

    return X, V, Y


def split_training_data(image_pairs, train_ratio=0.70, val_ratio=0.20, seed=42, print_stats=True):
    data = image_pairs.copy()

    # Random shuffle zodat stains door elkaar zitten
    random.seed(seed)
    random.shuffle(data)

    length = len(data)
    length_train = int(length * train_ratio)
    length_val = int(length * val_ratio)

    X = data[:length_train]
    V = data[length_train:length_train + length_val]
    Y = data[length_train + length_val:]

    if print_stats:
        def stain_counts(split):
            return Counter([item[2] for item in split])

        print("Total samples:", length)
        print("Train samples:", len(X), stain_counts(X))
        print("Val samples:", len(V), stain_counts(V))
        print("Test samples:", len(Y), stain_counts(Y))

    return X, V, Y

def get_frame_labels():
    directories = []
    frame_label_dict = {}

    for subdir, dirs, files in os.walk('../us_data'):
        if len(subdir.split("/")) == 2:
            directories.append(dirs)

    directories = directories[0]
    directories = sorted(list(set(directories)))

    for directory in directories:
        csv_file_path = f'../us_data/{directory}/frame_label.csv'
        image_folder_path = f'../us_data/{directory}/mask_enhance'

        frame_ids = {}
        id_3, id_4, id_5, id_6 = [], [], [], []
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Skip the header row
            for row in csv_reader:
                frame_id = row[1]
                frame_label = row[2]
                if frame_label == '3':
                    id_3.append(frame_id)
                elif frame_label == '4':
                    id_4.append(frame_id)
                elif frame_label == '5':
                    id_5.append(frame_id)
                elif frame_label == '6':
                    id_6.append(frame_id)
            
            frame_ids['3'] = id_3
            frame_ids['4'] = id_4
            frame_ids['5'] = id_5
            frame_ids['6'] = id_6
            frame_label_dict[directory] = frame_ids

    return frame_label_dict


################################################################################################### NIEUW
def identify_files(files):
    image_file = None
    instance_file = None
    seg_file = None

    for f in sorted(files):
        name = f.lower()

        if "instancemask" in name:
            instance_file = f
        elif "mask" in name and "instancemask" not in name:
            seg_file = f
        elif name.endswith("-img.tif") or name.endswith("-img.tiff"):
            image_file = f

    return image_file, instance_file, seg_file


def read_histopathology_data(root, image_size=192):

    data = []

    for stain_folder in os.listdir(root):

            # tijdelijk mIF overslaan
        if stain_folder == "mIF":
            print("Skipping mIF for baseline debugging", flush=True)
            continue

        stain_path = os.path.join(root, stain_folder)

        if not os.path.isdir(stain_path):
            continue

        for sample_folder in os.listdir(stain_path):

            sample_path = os.path.join(stain_path, sample_folder)

            if not os.path.isdir(sample_path):
                continue

            files = [
                f for f in os.listdir(sample_path)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                )
            ]

            image_file, _, seg_file = identify_files(files)

            if image_file is None or seg_file is None:
                continue

            image_path = os.path.join(sample_path, image_file)
            mask_path = os.path.join(sample_path, seg_file)

            img_pil = Image.open(image_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)
            mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)

            img = np.array(img_pil, dtype=np.float32) / 255.0
            mask_raw = np.array(mask_pil, dtype=np.float32)

            # Skip blank/corrupted images
            if img.max() < 0.01 or img.std() < 0.01:
                print(f"Skipping blank image: {image_path}", flush=True)
                continue

            if stain_folder == "DAPI" and img.mean() > 0.85 and img.std() < 0.15:
                print(f"Skipping low-contrast bright DAPI image: {image_path}")
                continue

            if stain_folder == "mIF":
                mask = (mask_raw < 128).astype(np.float32)
            else:
                mask = (mask_raw > 0).astype(np.float32)

            # data.append((img, mask, stain_folder))
            data.append((img, mask, stain_folder, sample_folder))
            

    print("Loaded histopathology samples:", len(data))

    return data


# def read_histopathology_data(root, image_size=192):

#     data = []

#     for stain_folder in os.listdir(root):

#         if stain_folder == "mIF":
#             print("Skipping mIF for baseline debugging", flush=True)
#             continue

#         stain_path = os.path.join(root, stain_folder)

#         if not os.path.isdir(stain_path):
#             continue

#         for sample_folder in os.listdir(stain_path):

#             sample_path = os.path.join(stain_path, sample_folder)

#             if not os.path.isdir(sample_path):
#                 continue

#             files = [
#                 f for f in os.listdir(sample_path)
#                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
#             ]

#             image_file, _, seg_file = identify_files(files)

#             if image_file is None or seg_file is None:
#                 continue

#             image_path = os.path.join(sample_path, image_file)
#             mask_path = os.path.join(sample_path, seg_file)

#             # Image veilig laden, ook voor 16-bit tif
#             img_pil = Image.open(image_path)
#             img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)
#             img = np.array(img_pil, dtype=np.float32)

#             Normalize op basis van echte range
#             if img.max() > 0:
#                 img = img / img.max()

#             # p1, p99 = np.percentile(img, (1, 99))
#             # img = (img - p1) / (p99 - p1 + 1e-6)
#             # img = np.clip(img, 0, 1)

#             # Als grayscale/16-bit single channel: maak RGB-achtig 3 channels
#             if img.ndim == 2:
#                 img = np.stack([img, img, img], axis=-1)

#             # Als RGBA of meer channels: pak eerste 3
#             if img.shape[-1] > 3:
#                 img = img[:, :, :3]

#             img = img.astype(np.float32)

#             # Mask laden
#             mask_pil = Image.open(mask_path).convert("L")
#             mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)
#             mask_raw = np.array(mask_pil, dtype=np.float32)

#             # Skip blank/corrupted images
#             if img.max() < 0.01 or img.std() < 0.01:
#                 print(f"Skipping blank image: {image_path}", flush=True)
#                 continue

#             if stain_folder == "mIF":
#                 mask = (mask_raw < 128).astype(np.float32)
#             else:
#                 mask = (mask_raw > 0).astype(np.float32)

#             data.append((img, mask, stain_folder, sample_folder))

#     print("Loaded histopathology samples:", len(data))

#     return data



def read_image_mask_folder_dataset(root, stain_name, image_size=192):
    from pathlib import Path
    from PIL import Image
    import numpy as np

    root = Path(root)
    image_dir = root / "images"
    mask_dir = root / "masks"

    data = []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            continue

        stem = img_path.stem

        possible_masks = [
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}_mask.png",
            mask_dir / f"{stem}.tif",
            mask_dir / f"{stem}.tiff",
        ]

        mask_path = next((p for p in possible_masks if p.exists()), None)

        if mask_path is None:
            print("No mask for:", img_path.name)
            continue

        img = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)

        data.append((img, mask, stain_name, img_path.stem))

    print(f"Loaded {stain_name}: {len(data)}")
    return data



def read_bbbc038_dataset(root, stain_name="BBBC038_DAPI", image_size=192):
    from pathlib import Path
    from PIL import Image
    import numpy as np

    root = Path(root)
    data = []

    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        image_dir = sample_dir / "images"
        mask_path = sample_dir / "combined_mask.png"

        if not image_dir.exists() or not mask_path.exists():
            continue

        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))

        if len(image_files) == 0:
            continue

        img_path = image_files[0]

        img = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)

        data.append((img, mask, stain_name, sample_dir.name))

    print(f"Loaded {stain_name}: {len(data)}")
    return data












def split_single_stain(image_pairs, stain_name="mIF", train_ratio=0.70, val_ratio=0.20, seed=42, print_stats=True):
    import random
    from collections import Counter

    data = [item for item in image_pairs if item[2] == stain_name]

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X = data[:n_train]
    V = data[n_train:n_train + n_val]
    Y = data[n_train + n_val:]

    if print_stats:
        print(f"Single-stain split: {stain_name}")
        print("Total:", n)
        print("Train:", len(X), Counter([item[2] for item in X]))
        print("Val:", len(V), Counter([item[2] for item in V]))
        print("Test:", len(Y), Counter([item[2] for item in Y]))

    return X, V, Y
########################################################################################################

# =============================================================================
# Split the data accordingly, patient wise
# =============================================================================

def get_data_jnu(label_dict, label='5', training=0.70, validation=0.20, testing=0.10):
    data_X, data_V, data_Y = [], [], []
    
    for image_name, label_data in label_dict.items():
        for label_id, frame_ids in label_data.items():
            if label_id == label:
                # Calculate split indices
                total_frames = len(frame_ids)
                training_len = int(total_frames * training)
                validation_len = int(total_frames * validation)

                # Split the frame_ids into training, validation, and testing
                train_ids = frame_ids[:training_len]
                val_ids = frame_ids[training_len:training_len + validation_len]
                test_ids = frame_ids[training_len + validation_len:]
                # Add training data
                for frame_id in train_ids:
                    data_X.append((f'../us_data/{image_name}/image/{image_name}_{frame_id}.png', f'../us_data/{image_name}/mask_enhance/{image_name}_{frame_id}_mask.png'))

                # Add validation data
                for frame_id in val_ids:
                    data_V.append((f'../us_data/{image_name}/image/{image_name}_{frame_id}.png', f'../us_data/{image_name}/mask_enhance/{image_name}_{frame_id}_mask.png'))

                # Add testing data
                for frame_id in test_ids:
                    data_Y.append((f'../us_data/{image_name}/image/{image_name}_{frame_id}.png', f'../us_data/{image_name}/mask_enhance/{image_name}_{frame_id}_mask.png'))

    return data_X, data_V, data_Y

# =============================================================================
# Read the images and resize them
# =============================================================================

def read_data_jnu(image_files: list, target_size=(256, 256)):
    images = []
    for image_file in image_files:
        if not isinstance(image_file, str):
            image1, image2 = image_file

            ############################################################################################### NIEUW
            # image = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
            # mask = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

            image = cv2.imread(image1, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

            ##############################################################################################

            resized_image = cv2.normalize(cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            resized_mask = cv2.normalize(cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0

            # numpy float 64
            resized_mask = (resized_mask > 0.5).astype(np.float32)

            images.append((resized_image, resized_mask))
        else:

            ################################################################################## NIUEW
            # image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # resized_image = cv2.normalize(cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            resized_image = resized_image.astype(np.float32) / 255.0
            ##################################################################################

            images.append(resized_image)
        
    return images

def reading_training_data_fetal(image_folder):
    image_pairs = []
    
    # List all the filenames
    filenames = os.listdir(image_folder)
    # Filter relevant filenames and sort them by the numeric part of the filename
    image_files = [filename for filename in filenames if filename.endswith(("_HC.png", "_2HC.png", "_3HC.png", "_4HC.png"))]
    
    # Sort files by extracting the number part before '_HC', '_2HC', etc.
    image_files.sort(key=lambda x: int(x.split('_')[0]))  # Assumes filename starts with a number (e.g., '430_HC.png')
    
    for filename in image_files:
        # Extract base name (e.g., "430" from "430_HC.png" or "431_2HC.png")
        if filename.endswith("_HC.png"):
            base_name = filename[:-7]
        elif filename.endswith("_2HC.png") or filename.endswith("_3HC.png") or filename.endswith("_4HC.png"):
            base_name = filename[:-8]
        
        # Create image path
        image_path = os.path.join(image_folder, filename)
        
        # Determine the correct mask path based on the suffix of the image
        if filename.endswith("_HC.png"):
            mask_path = os.path.join(image_folder, f"{base_name}_HC_Annotation.png")
        elif filename.endswith("_2HC.png"):
            mask_path = os.path.join(image_folder, f"{base_name}_2HC_Annotation.png")
        elif filename.endswith("_3HC.png"):
            mask_path = os.path.join(image_folder, f"{base_name}_3HC_Annotation.png")
        elif filename.endswith("_4HC.png"):
            mask_path = os.path.join(image_folder, f"{base_name}_4HC_Annotation.png")

        # Check if mask exists
        if os.path.exists(mask_path):
            # Read image and mask in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # ===== FILL THE CIRCLE IN THE MASK =====
            # Threshold to get binary mask (white circle on black background)
            _, binary_mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            
            # Find contours of the circle
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a filled mask
            filled_mask = np.zeros_like(mask)
            cv2.drawContours(filled_mask, contours, -1, 255, -1)  # Fill with white
            
            # Update the mask to the filled version
            mask = filled_mask
            # ===== END OF FILLING =====
            
            # Convert to float32 and normalize
            image, mask = image.astype(np.float32), mask.astype(np.float32)
            
            # Resize the image and mask
            image = cv2.normalize(cv2.resize(image, (192, 192), interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            mask = cv2.normalize(cv2.resize(mask, (192, 192), interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0

            # Add the image and mask pair to the list
            image_pairs.append((image, mask))
    print(len(image_pairs))
    return image_pairs

def reading_data_tg3k(total_images: int = 3584, test_split: float = 0.1, val_split: float = 0.1):
    # Read the data
    data_X, data_V, data_Y = [], [], []
    
    # Get all image and mask paths
    image_paths = sorted(glob("../tg3k/thyroid-image/*.jpg"))
    mask_paths = sorted(glob("../tg3k/thyroid-mask/*.jpg"))
    
    # Ensure we have matching pairs
    assert len(image_paths) == len(mask_paths), "Number of images and masks don't match"
    indices = list(range(len(image_paths)))
    random.seed(42)
    random.shuffle(indices)
    # Calculate split indices
    num_test = int(total_images * test_split)
    num_val = int(total_images * val_split)
    num_train = total_images - num_test - num_val
    
    for i, idx in enumerate(indices[:total_images]):
        try:
            # Read image and mask
            image = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Warning: Could not read image pair {idx+1}, skipping...")
                continue
                
            # Normalize image to [0,1]
            image = image.astype(np.float32)
            if np.max(image) > 0:
                image = image / 255.0  # Since it's JPG, max is 255
            
            # Normalize mask to [0,1] and binarize
            mask = mask.astype(np.float32)
            if np.max(mask) > 0:
                mask = mask / 255.0
            mask = np.where(mask > 0.5, 1.0, 0.0)  # Binarize
            
            # Resize to 192x192 if needed (assuming your original size is different)
            if image.shape != (192, 192):
                image = cv2.normalize(cv2.resize(image, (192, 192), interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
                mask = cv2.normalize(cv2.resize(mask, (192, 192), interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0
                mask = np.where(mask > 0.5, 1.0, 0.0)  # Re-binarize after resize
            
            # Split into train, validation, test
            if idx < num_test:
                data_Y.append((image, mask))
            elif idx < num_test + num_val:
                data_V.append((image, mask))
            else:
                data_X.append((image, mask))
                
        except Exception as e:
            print(f"Error processing image {idx+1}: {str(e)}")
            continue
    
    print(f"Loaded {len(data_X)} training, {len(data_V)} validation, {len(data_Y)} test samples")
    return data_X, data_V, data_Y


# =============================================================================
# Data Loading
# =============================================================================
def reading_camus_data(data_path="../CAMUS_public/database_nifti", image_size=192):
    data_X, data_V, data_Y = [], [], []

    name_list = os.listdir(data_path)
    random.seed(42)  # for reproducibility, or remove for different splits each run
    random.shuffle(name_list)   
    for idx, name in enumerate(name_list):
        for view in ['4CH', '2CH']:
            # Decide which files to load
            img_file = f"{name}/{name}_{view}_ED.nii.gz"
            msk_file = f"{name}/{name}_{view}_ED_gt.nii.gz"

            img = nib.load(os.path.join(data_path, img_file)).get_fdata()
            mask = nib.load(os.path.join(data_path, msk_file)).get_fdata()

            # Normalize mask to binary
            mask[mask != 1] = 0
            mask[mask > 0] = 1

            # Normalize image to [0,1]
            img = img.astype(np.float32)
            if np.max(img) > 0:
                img /= np.max(img)
            else:
                img = np.zeros_like(img, dtype=np.float32)

            # Resize both image and mask
            img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            # Normalize to [0,1] after resize
            img_resized = cv2.normalize(img_resized, None, 0, 1, cv2.NORM_MINMAX)
            mask_resized = cv2.normalize(mask_resized, None, 0, 1, cv2.NORM_MINMAX)
            mask_resized = np.clip(mask_resized, 0, 1)  # ensure binary

            if idx < int(0.6 * len(name_list)):  # Training (60%)
                data_X.append((img_resized, mask_resized))
            elif idx < int(0.8 * len(name_list)):  # Validation (20%)
                data_V.append((img_resized, mask_resized))
            else:  # Testing (20%)
                data_Y.append((img_resized, mask_resized))

    return data_X, data_V, data_Y

def reading_data(three_dimensional: int = 16):
    # Updated split: 10 for training, 3 for validation, 3 for testing
    data_X, data_V, data_Y = [], [], []

    for i in range(1, three_dimensional + 1):
        images = pydicom.dcmread(f"./thyroid/data/D{i:02d}.dcm")
        masks = pydicom.dcmread(f"./thyroid/groundtruth/D{i:02d}.dcm")
        
        for image, mask in zip(images.pixel_array, masks.pixel_array):
            # Normalize image to [0,1], ensuring no division by zero
            image, mask = image.astype(np.float32), mask.astype(np.float32)

            if np.max(image) > 0:
                image = image / np.max(image)
            else:
                image = np.zeros_like(image)

            if np.max(mask) > 0:
                mask = mask / np.max(mask)
            else:
                mask = np.zeros_like(mask)
            
            # Skip images where mask is completely black (no thyroid)
            if np.max(mask) == 0:
                continue  # Skip this image-mask pair

            image = cv2.normalize(cv2.resize(image, (192, 192), interpolation=cv2.INTER_LINEAR), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            mask = cv2.normalize(cv2.resize(mask, (192, 192), interpolation=cv2.INTER_NEAREST), None, 0, 255, cv2.NORM_MINMAX) / 255.0
            
            # Ensure mask is binary (0 or 1)
            mask = np.clip(mask, 0, 1)

            if i == 7 or i == 15:  # Training (D01-D10)
                data_Y.append((image, mask))
            elif i == 12 or i == 6:  # Validation (D11-D13)
                data_V.append((image, mask))
            else:  # Test (D14-D16)
                data_X.append((image, mask))

    return data_X, data_V, data_Y

def reading_busi_data(busi_root="./BUSI", image_size=192, sam2_results_dir="Results_BUSI_MEDSAM2"):
    """
    Returns:
        data_X, data_V, data_Y — lists of (image, mask, sam2_pred)
    """
    classes = ['benign', 'malignant','normal']
    #classes = ['normal']
    all_items = []

    for cls in classes:
        img_dir = os.path.join(busi_root, cls)
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png') and '_mask' not in f]

        for img_file in img_files:
            base = img_file[:-4]  # remove .png
            mask_candidates = [m for m in os.listdir(img_dir) if m.startswith(base + "_mask")]
            if not mask_candidates:
                continue

            mask_file = mask_candidates[0]
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(img_dir, mask_file)

            all_items.append((img_path, mask_path, cls))

    # Shuffle for splitting
    random.seed(42)
    random.shuffle(all_items)

    n = len(all_items)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_items = all_items[:n_train]
    val_items = all_items[n_train:n_train+n_val]
    test_items = all_items[n_train+n_val:]

    def read_and_process(items):
        data = []
        for img_path, mask_path, cls in items:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            # Normalize and resize
            img = img.astype(np.float32)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
            mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
            mask = (mask > 0.5).astype(np.float32)

            # SAM2 predictions (if available)
            sam2_sample = -1 * np.ones_like(mask)
            pred_filename = os.path.basename(img_path).replace('.png', '.png_prediction.jpg')
            pred_path = os.path.join(sam2_results_dir, pred_filename)

            if os.path.exists(pred_path):
                try:
                    sam2_sample = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                    sam2_sample = cv2.resize(sam2_sample, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                    sam2_sample = cv2.normalize(sam2_sample, None, 0, 1, cv2.NORM_MINMAX)
                    sam2_sample = (sam2_sample > 0.5).astype(np.float32)
                except:
                    sam2_sample = -1 * np.ones_like(mask)

            data.append((img, mask, sam2_sample))

        return data

    data_X = read_and_process(train_items)
    data_V = read_and_process(val_items)
    data_Y = read_and_process(test_items)

    print(f"BUSI: {len(data_X)} train, {len(data_V)} val, {len(data_Y)} test samples.")
    return data_X, data_V, data_Y


def reading_busbra(busbra_root="./BUSBRA", image_size=192, sam2_results_dir="Results_BUSBRA_MEDSAM2"):
    """
    Reads the BUSBRA dataset and returns:
        data_X, data_V, data_Y — lists of (image, mask, sam2_pred)

    Expected structure:
        BUSBRA/
            ├── Images/
            ├── Masks/
            └── bus_data.csv   # columns: 'ID', 'Pathology'
    """
    csv_path = os.path.join(busbra_root, "bus_data.csv")
    df = pd.read_csv(csv_path)
    # Expect columns: 'ID', 'Pathology' (adjust if different)
    filename_to_class = dict(zip(df['ID'], df['Pathology']))

    img_dir = os.path.join(busbra_root, "Images")
    mask_dir = os.path.join(busbra_root, "Masks")

    classes = ['benign', 'malignant']
    all_items = []

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    for img_file in img_files:
        img_id = img_file[:-4]  # remove .png
        cls = filename_to_class.get(img_id, None)
        if cls not in classes:
            continue

        base = img_file[4:-4]  # remove prefix like "img_" if exists
        mask_file = f"mask_{base}.png"
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            continue

        img_path = os.path.join(img_dir, img_file)
        all_items.append((img_path, mask_path, cls))

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_items)

    n = len(all_items)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_items = all_items[:n_train]
    val_items = all_items[n_train:n_train + n_val]
    test_items = all_items[n_train + n_val:]

    def read_and_process(items):
        data = []
        for img_path, mask_path, cls in items:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            # Resize + normalize
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            img = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            mask = cv2.normalize(mask.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            mask = (mask > 0.5).astype(np.float32)

            # Load SAM2 prediction if available
            sam2_sample = -1 * np.ones_like(mask)
            pred_filename = os.path.basename(img_path) + '_prediction.jpg'
            pred_path = os.path.join(sam2_results_dir, pred_filename)

            if os.path.exists(pred_path):
                try:
                    sam2_sample = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                    sam2_sample = cv2.resize(sam2_sample, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                    sam2_sample = cv2.normalize(sam2_sample, None, 0, 1, cv2.NORM_MINMAX)
                    sam2_sample = (sam2_sample > 0.5).astype(np.float32)
                except:
                    sam2_sample = -1 * np.ones_like(mask)

            data.append((img, mask, sam2_sample))

        return data

    data_X = read_and_process(train_items)
    data_V = read_and_process(val_items)
    data_Y = read_and_process(test_items)

    print(f"BUSBRA: {len(data_X)} train, {len(data_V)} val, {len(data_Y)} test samples.")
    return data_X, data_V, data_Y

        
