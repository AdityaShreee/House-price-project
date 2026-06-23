# House Price Prediction — ML Project (S3 Deployment)

A complete machine learning project that predicts median house values using
the **California Housing dataset** (the standard modern replacement for the
deprecated Boston Housing dataset). The trained model artifacts are uploaded
to an **AWS S3 bucket** for storage/deployment.

## Project Structure

```
house_price_project/
├── data/
│   └── housing.csv              # dataset (generated on first run)
├── models/
│   ├── house_price_model.pkl    # trained model (best of 5 candidates)
│   ├── scaler.pkl                # StandardScaler used for preprocessing
│   └── metadata.json             # feature list, metrics, model name
├── outputs/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── model_comparison.csv
├── train_model.py                # full training pipeline
├── predict.py                    # load model + run inference
├── upload_to_s3.py               # uploads artifacts to S3
├── requirements.txt
└── README.md
```

## What the pipeline does (`train_model.py`)

1. Loads the California Housing dataset (8 features: median income, house
   age, average rooms, average bedrooms, population, average occupancy,
   latitude, longitude → target: median house value).
2. Generates EDA plots (target distribution, correlation heatmap).
3. Splits data 80/20 and scales features with `StandardScaler`.
4. Trains and compares 5 models: Linear Regression, Ridge, Decision Tree,
   Random Forest, Gradient Boosting.
5. Picks the best model by RMSE and saves it + the scaler + metadata as
   `.pkl` / `.json` files — these are exactly what gets uploaded to S3.

On a test run, **Gradient Boosting** came out on top with R² ≈ 0.79
(your numbers will vary slightly depending on the data and random seed).

## Running it locally

```bash
pip install -r requirements.txt
python train_model.py      # trains everything, saves models/
python predict.py          # example prediction using the saved model
```

> Note: `train_model.py` calls `fetch_california_housing()`, which downloads
> the dataset from sklearn's servers the first time. If you're behind a
> restrictive network/firewall, the script automatically falls back to a
> synthetic dataset with the same schema, so the pipeline still runs.

---

## Deploying to AWS S3 — Step by Step

S3 is **object storage**, not a compute service — it can't "run" your model.
What you're deploying here is the trained model file (and dataset) so it's
stored centrally, versioned, and can be pulled by other services (a
Lambda function, EC2, SageMaker, or another app) whenever they need to make
predictions.

### Step 1 — Create an AWS account & IAM user
1. Sign in to the [AWS Console](https://aws.amazon.com/console/).
2. Go to **IAM → Users → Add User**.
3. Give it **programmatic access** and attach the policy
   `AmazonS3FullAccess` (or a scoped custom policy for production).
4. Save the **Access Key ID** and **Secret Access Key** — you'll need them
   in Step 3.

### Step 2 — Install and configure the AWS CLI
```bash
pip install awscli boto3
aws configure
```
You'll be prompted for:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g. `ap-south-1` for Mumbai)
- Default output format (`json` is fine)

### Step 3 — Create an S3 bucket
Either via console (**S3 → Create bucket**, pick a globally unique name),
or via CLI:
```bash
aws s3 mb s3://your-house-price-bucket --region ap-south-1
```
Bucket names must be globally unique across *all* AWS accounts, lowercase,
no underscores.

### Step 4 — Upload your model artifacts
Option A — using the provided script (edit `BUCKET_NAME` and `REGION` at
the top of `upload_to_s3.py` first):
```bash
python upload_to_s3.py
```

Option B — directly via CLI:
```bash
aws s3 cp models/house_price_model.pkl s3://your-house-price-bucket/house-price-model/
aws s3 cp models/scaler.pkl            s3://your-house-price-bucket/house-price-model/
aws s3 cp models/metadata.json         s3://your-house-price-bucket/house-price-model/
aws s3 cp data/housing.csv             s3://your-house-price-bucket/house-price-model/
```

### Step 5 — Verify the upload
```bash
aws s3 ls s3://your-house-price-bucket/house-price-model/
```
Or check visually in the S3 console under your bucket.

### Step 6 — (Optional) Set bucket permissions
By default, new buckets block all public access — keep it that way unless
you specifically need public read access. For a private project, leave
**Block all public access** ON, and instead grant access via IAM roles to
whichever service (Lambda, EC2, SageMaker) needs to download the model.

If you ever need a specific object to be publicly downloadable (e.g. for a
demo), you can attach a bucket policy scoped to just that prefix — avoid
making the whole bucket public.

### Step 7 — Pulling the model back down (when you need to use it)
From any EC2 instance, Lambda function, or local machine with the right IAM
permissions:
```bash
aws s3 cp s3://your-house-price-bucket/house-price-model/house_price_model.pkl ./models/
aws s3 cp s3://your-house-price-bucket/house-price-model/scaler.pkl ./models/
```
Then run `predict.py` as normal — it loads from the local `models/` folder,
which now contains the files pulled from S3.

---

## Going further (optional next steps)
- **Lambda + API Gateway**: wrap `predict.py`'s logic in a Lambda function
  that loads the model from S3 on cold start, so you get a live prediction
  API without managing a server.
- **SageMaker**: host the model as a managed real-time endpoint instead of
  S3-only storage, if you need low-latency live inference at scale.
- **Versioning**: enable S3 bucket versioning so retrained models don't
  overwrite older ones silently.
