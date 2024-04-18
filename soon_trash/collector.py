import tarfile

# Extract the tar file
tar = tarfile.open('reuters21578.tar.gz', 'r:gz')

# Extract all the files
tar.extractall()

# Close the tar file
tar.close()

