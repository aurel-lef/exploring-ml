Param(
[parameter(Mandatory=$true)] [string] $project_id,
[parameter(Mandatory=$true)] [string] $region,
[parameter(Mandatory=$true)] [string] $bucket,
[parameter(Mandatory=$true)] [string] $trainDatasetLocalPath,
[parameter(Mandatory=$true)] [string] $testDatasetLocalPath,
[parameter(Mandatory=$true)] [string] $cluster,
[parameter(Mandatory=$true)] [string] $scriptPath,
[parameter()] [string[]] $modulePaths
)

function CreateBucketIfNotExists{
    If((gsutil ls) -notcontains $bucket)
    {
        Write-Host "creating $bucket"
        gsutil mb -p $project_id -l $region $bucket
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
    --project=$project_id `
    --region=$region

    if($clusters -notcontains $cluster)
    {
        Write-Host "creating cluster $cluster"
        gcloud dataproc clusters create $cluster `
        --project=$project_id `
        --region=$region `
        --image-version=preview `
        --master-machine-type n1-standard-4 `
        --master-boot-disk-size 30GB `
        --num-workers 2 `
        --worker-machine-type n1-standard-4 `
        --worker-boot-disk-size 30GB `
        --max-idle=30m `
        --max-age=1d
    }
}

function DeleteCluster{
    Write-Host "deleting cluster $cluster"
    gcloud dataproc clusters delete $cluster `
    --project=$project_id `
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
        -- $gsTrainDataset $gsTestDataset "$($bucket)pyspark-gcp-dataproc-prediction"
}



# add also project creation and api enabling

# add trailing "/" in the bucket name
If($bucket[-1] -ne "/")
{
    $bucket = "$bucket/"
}


CreateBucketIfNotExists
CopyFileToBucket -inputPath $trainDatasetLocalPath
CopyFileToBucket -inputPath $testDatasetLocalPath
CreateClusterIfNotExists
SendJob
DeleteCluster

