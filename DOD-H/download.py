import boto3
import os

def download_s3_files(bucket_name, local_directory):
    # Create an S3 client
    s3 = boto3.client('s3')

    # List all objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)['Contents']
    print('///////// objects ////////////// \n')
    print(objects)
    print('\n\n\n\ ///////// number ////////////// \n')
    print(f'number of files {len(objects)}')

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
        
    files = os.listdir(local_directory)
    
    # Download each file
    for obj in objects:
        key = obj['Key']
        cond = True
        local_file_path = os.path.join(local_directory, key)
        s3.download_file(bucket_name, key, local_file_path)
        print(f'Downloaded: {key} to {local_file_path}')

if __name__ == "__main__":
    bucket_name = 'dreem-dod-h'
    local_directory = ''

    download_s3_files(bucket_name, local_directory)
