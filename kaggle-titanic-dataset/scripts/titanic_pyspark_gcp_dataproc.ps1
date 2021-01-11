Param(
[parameter(Mandatory=$true)] [string] $projectId,
[parameter(Mandatory=$true)] [string] $region,
[parameter(Mandatory=$true)] [string] $bucket,
[parameter(Mandatory=$true)] [string] $trainDatasetLocalPath,
[parameter(Mandatory=$true)] [string] $testDatasetLocalPath,
[parameter(Mandatory=$true)] [string] $cluster,
[parameter(Mandatory=$true)] [string] $scriptPath,
[parameter(Mandatory=$true)] [bool] $hyperparameterTuning,
[parameter()] [string[]] $modulePaths
)

function CreateBucketIfNotExists{
    If((gsutil ls) -notcontains $bucket)
    {
        Write-Host "creating $bucket"
        gsutil mb -p $projectId -l $region $bucket
    }
    else
    {
        Write-Host "bucket $bucket already exists"
    }
}

function CopyFileToBucket{
    Param(
        [parameter(Mandatory=$true)] [string] $inputPath
    )

    $filesInBucket = gsutil ls $bucket

    $gsFile = GetGSPath -inputPath $inputPath

    Write-Host "copying $inputPath to $bucket"
    gsutil cp $inputPath $bucket
}

function GetGSPath{
    Param(
        [parameter(Mandatory=$true)] [string] $inputPath
    )

    Write-Host "$gsFile"
    $fileName = Split-Path $inputPath -leaf
    return "$($bucket)$($fileName)"
}

function CreateClusterIfNotExists{
    $clusters = gcloud dataproc clusters list `
    --project=$projectId `
    --region=$region

    if($clusters -notcontains $cluster)
    {
        Write-Host "creating cluster $cluster"
        gcloud dataproc clusters create $cluster `
        --project=$projectId `
        --region=$region `
        --image-version=preview `
        --master-machine-type n1-standard-4 `
        --master-boot-disk-size 1TB `
        --num-workers 5 `
        --worker-machine-type n1-standard-4 `
        --worker-boot-disk-size 500GB `
        --enable-component-gateway `
        --max-idle=30m `
        --max-age=1d
    }
}

function DeleteCluster{
    Write-Host "deleting cluster $cluster"
    gcloud dataproc clusters delete $cluster `
    --project=$projectId `
    --region=$region `
    --quiet
}

function SendJob{
    $gsTrainDataset = GetGSPath -inputPath $trainDatasetLocalPath
    $gsTestDataset = GetGSPath -inputPath $testDatasetLocalPath
    Write-Host "sending Job $scriptPath with traindataset=$gsTrainDataset to cluster $cluster"
    gcloud dataproc jobs submit pyspark $scriptPath `
        --cluster=$cluster `
        --region=$region `
        --py-files $modulePaths `
        -- $gsTrainDataset $gsTestDataset "$($bucket)pyspark-gcp-dataproc-prediction" $hyperparameterTuning
}

function CreateProject{
    $projectIds = gcloud projects list
    if($projectIds -notcontains $projectId)
    {
        Write-Host "creating project $projectId"
        gcloud projects create $projectId        gcloud config set project $projectId
        gcloud services enable storage-api.googleapis.com
        gcloud services enable dataproc.googleapis.com
        gcloud services enable compute.googleapis.com
    }
    else{
        Write-Host "project $projectId already exists"
    }
}


# add trailing "/" in the bucket name
If($bucket[-1] -ne "/")
{
    $bucket = "$bucket/"
}

#CreateProject
CreateBucketIfNotExists
CopyFileToBucket -inputPath $trainDatasetLocalPath
CopyFileToBucket -inputPath $testDatasetLocalPath
CreateClusterIfNotExists
SendJob
DeleteCluster
# keep project

