import pandas as pd

from gretel_client.navigator_client import Gretel

datasets = ["adult", "beijing", "default", "diabetes", "magic", "news", "shoppers"]
gretel = Gretel(
    api_key="prompt",
    default_project_id="tabdiff-evaluate"
)


def run_gretel_eval(dataset_name):
    print(f"Running Gretel evaluation for {dataset_name}")

    train_data_path = f"/home/ubuntu/TabDiff/data/{dataset_name}/train.csv"
    test_data_path = f"/home/ubuntu/TabDiff/data/{dataset_name}/test.csv"
    synthetic_data_path = f"/home/ubuntu/TabDiff/tabdiff/result/{dataset_name}/learnable_schedule/2000/samples.csv"

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    synthetic_df = pd.read_csv(synthetic_data_path)
    if dataset_name in ("adult", "diabetes"):
        train_df = train_df.sample(20000)
        test_df = test_df.sample(1000)
        synthetic_df = synthetic_df.sample(20000)

    # Convert any Pandas Data Frames to Datasets
    training_file = gretel.files.upload(train_df, purpose="dataset")
    holdout_file = gretel.files.upload(test_df, purpose="dataset")
    synthetic_file = gretel.files.upload(synthetic_df, purpose="dataset")

    workflow = gretel.workflows.builder()
    try:
        print(workflow.id)
    except:
        pass

    workflow.add_step(gretel.tasks.Holdout(), [training_file.id, holdout_file.id], step_name="holdout")
    workflow.add_step(gretel.tasks.EvaluateSafeSyntheticsDataset(), [synthetic_file.id, "holdout"])

    results = workflow.run(wait_until_done=False)
    

for dataset in datasets:
    run_gretel_eval(dataset)
