gcloud builds submit --tag gcr.io/explore-ml-pyspark/app --timeout=20m

gcloud run deploy webapi --image gcr.io/explore-ml-pyspark/app --platform managed --region us-central1 --allow-unauthenticated