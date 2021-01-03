Param(
[parameter(Mandatory=$true)] [string] $project_id,
[parameter(Mandatory=$true)] [string] $region,
[parameter(Mandatory=$true)] [string] $bucket,
[parameter(Mandatory=$true)] [string] $trainDatasetLocalPath,
[parameter(Mandatory=$true)] [string] $testDatasetLocalPath,
[parameter(Mandatory=$true)] [string] $cluster,
[parameter(Mandatory=$true)] [string] $scriptPath
)

function CreateBucket{
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
    [parameter(Mandatory=$true)] [string] $inputPath

    $filesInBucket = gsutil ls $bucket

    $fileName = Split-Path $inputPath -leaf
    $gsFile = "$bucket$$fileName"
    If($filesInBucket -notcontains $gsFile)
    {
        Write-Host "copying $inputPath to $bucket"
        gsutil cp $inputPath $bucket
    }
}

function GetGSPath{
}

function CreateCluster{
    $clusters = gcloud dataproc clusters list `
    --project=$project_id `
    --region=$region

    if($clusters -notcontains $cluster)
    {
        Write-Host "creating cluster $cluster"
        gcloud dataproc clusters create $cluster `
        --project=$project_id `
        --region=$region `
        --single-node
    }
}

function DeleteCluster{
    gcloud dataproc clusters delete $cluster `
    --project=$project_id `
    --region=$region
}

function SendJob{

    gcloud dataproc jobs submit pyspark $scriptPath `
        --cluster=$cluster `
        --region=$region `
        -- $bucket/train.csv/
}



# add also project creation and api enabling

# add trailing "/" in the bucket name
If($bucket[-1] -ne "/")
{
    $bucket = "$bucket/"
}

# create 

CreateBucket
CopyFileToBucket -inputPath $trainDatasetLocalPath
CopyFileToBucket -inputPath $testDatasetLocalPath
#CreateCluster
SendJob
DeleteCluster

