from datasets import load_dataset,concatenate_datasets, DatasetDict
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--target_repo", type=str, default=None)
parser.add_argument("--private_repo", action="store_true", default=False)
args = parser.parse_args()
BINARIZE_LABELS = True
TARGET_HF_REPO = args.target_repo
PRIVATE_REPO = args.private_repo



def prep_jailbreak_and_alert():
    jailbreak_train = load_dataset("allenai/wildjailbreak", 'train', delimiter="\t", keep_default_na=False,)

    # Load the WildJailbreak evaluation set
    jailbreak_eval = load_dataset("allenai/wildjailbreak", "eval", split='train', delimiter="\t", keep_default_na=False,N)


    alert_vanilla = load_dataset("Babelscape/ALERT", "alert")
    alert_adversarial = load_dataset("Babelscape/ALERT", "alert_adversarial")

    def is_valid_vanilla_example(example):
        # Check if the example meets the criteria for inclusion
        return (
            example['vanilla'] and
            len(example['vanilla'].strip()) > 0 and
            'vanilla' in example['data_type']
        )

    def is_valid_adversarial_example(example):
        return (
            example['adversarial'] and
            len(example['adversarial'].strip()) > 0 and
            'adversarial' in example['data_type']
        )

    def map_to_vanilla_example(example):
        # Extract only the required fields
        return {
            'text': example['vanilla'],
            'label': example['data_type']
        }

    def map_to_adversarial_example(example):
        return {
            'text': example['adversarial'],
            'label': example['data_type']
        }

    # Filter and process the dataset
    def process_dataset(dataset, valid_fn, map_fn, is_eval: bool = False):
        # Step 1: Filter out invalid examples
        filtered_dataset = dataset.filter(valid_fn)
        
        # Step 2: Map to the desired structure
        processed_dataset = filtered_dataset.map(
            map_fn,
            remove_columns=['vanilla', 'adversarial', 'completion', 'data_type'] if not is_eval else ['adversarial', 'data_type'],  # Drop unnecessary columns
        )
        
        return processed_dataset

    vanilla_train_pool = process_dataset(jailbreak_train['train'], is_valid_vanilla_example, map_to_vanilla_example)
    adversarial_train_pool = process_dataset(jailbreak_train['train'], is_valid_adversarial_example, map_to_adversarial_example)


    wildjailbreak_test = process_dataset(jailbreak_eval, is_valid_adversarial_example, map_to_adversarial_example, is_eval=True)

    alert_vanilla = load_dataset("Babelscape/ALERT", "alert")
    alert_adversarial = load_dataset("Babelscape/ALERT", "alert_adversarial")


    alert_vanilla = alert_vanilla.rename_column("prompt", "text")

    alert_adversarial = alert_adversarial.rename_column("prompt", "text")


    # Get unique categories and their counts for vanilla dataset
    vanilla_categories = alert_vanilla["test"].unique("category")
    vanilla_category_counts = {cat: len(alert_vanilla["test"].filter(lambda x: x["category"] == cat)) for cat in vanilla_categories}

    # Initialize empty lists for train and test indices
    vanilla_train_indices = []
    vanilla_test_indices = []

    # Split each category roughly in half while maintaining proportions
    for category in vanilla_categories:
        category_indices = [i for i, x in enumerate(alert_vanilla["test"]) if x["category"] == category]
        half_size = len(category_indices) // 2
        
        # Shuffle indices for this category
        shuffled_indices = category_indices.copy()
        random.seed(42)
        random.shuffle(shuffled_indices)
        
        vanilla_train_indices.extend(shuffled_indices[:half_size])
        vanilla_test_indices.extend(shuffled_indices[half_size:half_size*2])

    # Create stratified train/test splits for vanilla
    alert_vanilla_train = alert_vanilla["test"].select(vanilla_train_indices)
    alert_vanilla_test = alert_vanilla["test"].select(vanilla_test_indices)

    # Remove any overlapping examples between train and test based on text
    alert_vanilla_test_texts = set(alert_vanilla_test["text"])
    alert_vanilla_train = alert_vanilla_train.filter(lambda x: x["text"] not in alert_vanilla_test_texts)

    # Get matching amount of harmful examples from wildjailbreak for training
    wildjailbreak_harmful = vanilla_train_pool.filter(lambda x: x["label"] == "vanilla_harmful").shuffle(seed=42)
    wildjailbreak_harmful_sample = wildjailbreak_harmful.select(range(len(alert_vanilla_train)))

    # Get matching amount of benign examples from wildjailbreak for training
    wildjailbreak_benign = vanilla_train_pool.filter(lambda x: x["label"] == "vanilla_benign").shuffle(seed=42)
    wildjailbreak_benign_sample = wildjailbreak_benign.select(range(len(alert_vanilla_train) * 2))

    # Get benign examples for alert_vanilla_test from remaining wildjailbreak benign examples
    remaining_wildjailbreak_benign = wildjailbreak_benign.select(range(len(alert_vanilla_train) * 2, len(wildjailbreak_benign)))
    alert_vanilla_test_benign = remaining_wildjailbreak_benign.select(range(len(alert_vanilla_test)))

    # Do the same stratified split for adversarial dataset
    adversarial_categories = alert_adversarial["test"].unique("category")
    adversarial_category_counts = {cat: len(alert_adversarial["test"].filter(lambda x: x["category"] == cat)) for cat in adversarial_categories}

    adversarial_train_indices = []
    adversarial_test_indices = []

    for category in adversarial_categories:
        category_indices = [i for i, x in enumerate(alert_adversarial["test"]) if x["category"] == category]
        half_size = len(category_indices) // 2
        
        shuffled_indices = category_indices.copy()
        random.seed(42)
        random.shuffle(shuffled_indices)
        
        adversarial_train_indices.extend(shuffled_indices[:half_size])
        adversarial_test_indices.extend(shuffled_indices[half_size:half_size*2])

    # Create stratified train/test splits for adversarial
    alert_adversarial_train = alert_adversarial["test"].select(adversarial_train_indices)
    alert_adversarial_test = alert_adversarial["test"].select(adversarial_test_indices)

    # Remove any overlapping examples between train and test based on text
    alert_adversarial_test_texts = set(alert_adversarial_test["text"])
    alert_adversarial_train = alert_adversarial_train.filter(lambda x: x["text"] not in alert_adversarial_test_texts)

    # Get matching amount of harmful examples from wildjailbreak for training
    wildjailbreak_adv_harmful = adversarial_train_pool.filter(lambda x: x["label"] == "adversarial_harmful").shuffle(seed=42)
    wildjailbreak_adv_harmful_sample = wildjailbreak_adv_harmful.select(range(len(alert_adversarial_train)))

    # Get matching amount of benign examples from wildjailbreak for training
    wildjailbreak_adv_benign = adversarial_train_pool.filter(lambda x: x["label"] == "adversarial_benign").shuffle(seed=42)
    wildjailbreak_adv_benign_sample = wildjailbreak_adv_benign.select(range(len(alert_adversarial_train) * 2))

    # Get benign examples for alert_adversarial_test from remaining wildjailbreak adversarial benign examples
    remaining_wildjailbreak_adv_benign = wildjailbreak_adv_benign.select(range(len(alert_adversarial_train) * 2, len(wildjailbreak_adv_benign)))
    alert_adversarial_test_benign = remaining_wildjailbreak_adv_benign.select(range(len(alert_adversarial_test)))

    # Get benign examples for wildjailbreak test from remaining benign examples
    remaining_vanilla_benign = wildjailbreak_benign.select(range(len(alert_vanilla_train) * 2 + len(alert_vanilla_test), len(wildjailbreak_benign)))
    remaining_adv_benign = wildjailbreak_adv_benign.select(range(len(alert_adversarial_train) * 2 + len(alert_adversarial_test), len(wildjailbreak_adv_benign)))

    wildjailbreak_test_vanilla_benign = remaining_vanilla_benign.select(range(len(wildjailbreak_test) // 2))
    wildjailbreak_test_adv_benign = remaining_adv_benign.select(range(len(wildjailbreak_test) // 2))

    # Combine harmful and benign examples for test sets
    alert_vanilla_test = concatenate_datasets([alert_vanilla_test, alert_vanilla_test_benign])
    alert_adversarial_test = concatenate_datasets([alert_adversarial_test, alert_adversarial_test_benign])
    wildjailbreak_test = concatenate_datasets([wildjailbreak_test, wildjailbreak_test_vanilla_benign, wildjailbreak_test_adv_benign])


    print("Alert Vanilla Train Size:", len(alert_vanilla_train))
    print("Alert Vanilla Test Size:", len(alert_vanilla_test))
    print("WildJailbreak Harmful Sample Size:", len(wildjailbreak_harmful_sample))
    print("WildJailbreak Benign Sample Size:", len(wildjailbreak_benign_sample))

    print("\nAlert Adversarial Train Size:", len(alert_adversarial_train))
    print("Alert Adversarial Test Size:", len(alert_adversarial_test))
    print("WildJailbreak Adversarial Harmful Sample Size:", len(wildjailbreak_adv_harmful_sample))
    print("WildJailbreak Adversarial Benign Sample Size:", len(wildjailbreak_adv_benign_sample))


    # Rename columns in ALERT datasets to match WildJailbreak format
    # Update labels for ALERT vanilla datasets
    alert_vanilla_train = alert_vanilla_train.map(lambda x: {"label": "vanilla_harmful"})
    alert_vanilla_test = alert_vanilla_test.map(lambda x: {"label": "vanilla_harmful"})

    # Update labels for ALERT adversarial datasets 
    alert_adversarial_train = alert_adversarial_train.map(lambda x: {"label": "adversarial_harmful"})
    alert_adversarial_test = alert_adversarial_test.map(lambda x: {"label": "adversarial_harmful"})

    # Print unique labels for each dataset
    print("Alert Vanilla Train Labels:", set(alert_vanilla_train["label"]))
    print("Alert Vanilla Test Labels:", set(alert_vanilla_test["label"]))
    print("WildJailbreak Harmful Sample Labels:", set(wildjailbreak_harmful_sample["label"]))
    print("WildJailbreak Benign Sample Labels:", set(wildjailbreak_benign_sample["label"]))

    print("\nAlert Adversarial Train Labels:", set(alert_adversarial_train["label"]))
    print("Alert Adversarial Test Labels:", set(alert_adversarial_test["label"])) 
    print("WildJailbreak Adversarial Harmful Sample Labels:", set(wildjailbreak_adv_harmful_sample["label"]))
    print("WildJailbreak Adversarial Benign Sample Labels:", set(wildjailbreak_adv_benign_sample["label"]))
    train_dataset = DatasetDict({
        "train": concatenate_datasets([
            alert_vanilla_train,
            alert_adversarial_train,
            wildjailbreak_harmful_sample,
            wildjailbreak_benign_sample,
            wildjailbreak_adv_harmful_sample, 
            wildjailbreak_adv_benign_sample
        ])
    })

    # Create test datasets in separate subsets
    test_datasets = DatasetDict({
        "wildjailbreak": DatasetDict({
            "test": wildjailbreak_test
        }),
        "alert_vanilla": DatasetDict({
            "test": alert_vanilla_test
        }),
        "alert_adversarial": DatasetDict({
            "test": alert_adversarial_test
        })
    })

    print("\nTrain dataset size:", len(train_dataset["train"]))
    print("WildJailbreak test size:", len(test_datasets["wildjailbreak"]["test"]))
    print("ALERT vanilla test size:", len(test_datasets["alert_vanilla"]["test"])) 
    print("ALERT adversarial test size:", len(test_datasets["alert_adversarial"]["test"]))

    # Combine all datasets into one for upload
    train_ds  = DatasetDict({
        "train": train_dataset["train"].remove_columns(["id", "attack_type"]),
    })
    test_wildjailbreak_ds = DatasetDict({
        "test": test_datasets["wildjailbreak"]["test"]
    })
    test_alert_vanilla_ds = DatasetDict({
        "test": test_datasets["alert_vanilla"]["test"].remove_columns(["id"])
    })
    test_alert_adversarial_ds = DatasetDict({
        "test": test_datasets["alert_adversarial"]["test"].remove_columns(["id", "attack_type"])
    })


    if BINARIZE_LABELS:
        def map_label(example):
            label = example["label"]
            return {"label": "harmful" if "harmful" in label.lower() else "benign"}

        train_ds["train"] = train_ds["train"].map(map_label)
        test_wildjailbreak_ds["test"] = test_wildjailbreak_ds["test"].map(map_label) 
        test_alert_vanilla_ds["test"] = test_alert_vanilla_ds["test"].map(map_label)
        test_alert_adversarial_ds["test"] = test_alert_adversarial_ds["test"].map(map_label)
        class2id = {}
        class2id['benign'] = 0
        class2id['harmful'] = 1

    else:
        train_classes = sorted(list(set(train_dataset["train"]["label"])))
        class2id = {class_: id for id, class_ in enumerate(train_classes)}



    train_ds["train"] = train_ds["train"].map(lambda x: {"label": class2id[x["label"]]})
    test_wildjailbreak_ds["test"] = test_wildjailbreak_ds["test"].map(lambda x: {"label": class2id[x["label"]]})
    test_alert_vanilla_ds["test"] = test_alert_vanilla_ds["test"].map(lambda x: {"label": class2id[x["label"]]})
    test_alert_adversarial_ds["test"] = test_alert_adversarial_ds["test"].map(lambda x: {"label": class2id[x["label"]]})


    # Clean up text formatting in all datasets
    def clean_text(example):
        text = example["text"]
        text = text.replace("\n", " ")
        text = text.replace("### Instruction:", "")
        text = text.replace("### Response:", "") 
        text = text.strip()
        return {"text": text}

    train_ds["train"] = train_ds["train"].map(clean_text)
    test_wildjailbreak_ds["test"] = test_wildjailbreak_ds["test"].map(clean_text)
    test_alert_vanilla_ds["test"] = test_alert_vanilla_ds["test"].map(clean_text)
    test_alert_adversarial_ds["test"] = test_alert_adversarial_ds["test"].map(clean_text)

    # Shuffle all datasets with seed 42
    train_ds["train"] = train_ds["train"].shuffle(seed=42)
    test_wildjailbreak_ds["test"] = test_wildjailbreak_ds["test"].shuffle(seed=42)
    test_alert_vanilla_ds["test"] = test_alert_vanilla_ds["test"].shuffle(seed=42)
    test_alert_adversarial_ds["test"] = test_alert_adversarial_ds["test"].shuffle(seed=42)

    # Remove category column if it exists
    if "category" in train_ds["train"].features:
        train_ds["train"] = train_ds["train"].remove_columns("category")
    if "category" in test_wildjailbreak_ds["test"].features:
        test_wildjailbreak_ds["test"] = test_wildjailbreak_ds["test"].remove_columns("category") 
    if "category" in test_alert_vanilla_ds["test"].features:
        test_alert_vanilla_ds["test"] = test_alert_vanilla_ds["test"].remove_columns("category")
    if "category" in test_alert_adversarial_ds["test"].features:
        test_alert_adversarial_ds["test"] = test_alert_adversarial_ds["test"].remove_columns("category")


    # Push to the Hub
    train_ds.push_to_hub(TARGET_HF_REPO, 'train', private=PRIVATE_REPO)
    test_wildjailbreak_ds.push_to_hub(TARGET_HF_REPO, 'test_wildjailbreak', private=PRIVATE_REPO)
    test_alert_vanilla_ds.push_to_hub(TARGET_HF_REPO, 'test_alert_vanilla', private=PRIVATE_REPO)
    test_alert_adversarial_ds.push_to_hub(TARGET_HF_REPO, 'test_alert_adversarial', private=PRIVATE_REPO)


def beaver_tails():
    beaver_tails = load_dataset("PKU-Alignment/BeaverTails", split='330k_test')
    def process_beaver_tails(example):
        return {
            "text": example["prompt"],
            "label": int(not example["is_safe"])  # Convert True->0, False->1
        }

    beaver_tails = beaver_tails.map(process_beaver_tails, remove_columns=["prompt", "response", "category", "is_safe"])
    test_beaver_tails_ds = DatasetDict({
        "test": beaver_tails
    })
    test_beaver_tails_ds.push_to_hub(TARGET_HF_REPO, 'test_beaver_tails', private=PRIVATE_REPO)

if __name__ == "__main__":
    prep_jailbreak_and_alert()
    beaver_tails()
