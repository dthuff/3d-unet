
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi

input_dir = "../data/multilabel/image/"
target_dir = "../data/multilabel/mask/"

patch_size = (16, 16, 16)
extraction_step = 16
num_classes = 6 #including background (e.g. num_classes must be >=2)
batch_size = 1
val_fraction = 0.2
epochs = 200

# Sort and display available data
input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".nii.gz")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".nii.gz") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:93], target_img_paths[:93]):
    print(input_path, "|", target_path)
