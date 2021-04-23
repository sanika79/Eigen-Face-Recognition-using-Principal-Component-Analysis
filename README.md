# Eigen-Face-Recognition-using-Principal-Component-Analysis 

M. A Turk and A. P. Pentland, “Face Recognition Using Eigenfaces", Proceedings of IEEE
CVPR 1991.

The paper puts forward a simple yet effective idea of using eigenfaces (obtained via PCA) to perform unsupervised face recognition.

Data: The following datasets consist of faces of 200 people and each person has two frontal
images (one with a neutral expression and the other with a smiling facial expression),
there are 400 full frontal face images manually registered and cropped.
http://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part1.zip
http://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part2.zip

Implementation and Experiments:
(a) Compute the principal components (PCs) using first 190 individuals’ neutral expression image. Plot the singular values of the data matrix and justify your choice of
principal components.
(b) Reconstruct one of 190 individuals’ neutral expression image using different number
of PCs. As you vary the number of PCs, plot the mean squared error (MSE) of
reconstruction versus the number of principal components to show the accuracy of
reconstruction. Comment on your result.
(c) Reconstruct one of 190 individuals’ smiling expression image using different number of PCs. Again, plot the MSE of reconstruction versus the number of principal
components and comment on your result.
(d) Reconstruct one of the other 10 individuals’ neutral expression image using different
number of PCs. Again, plot the MSE of reconstruction versus the number of principal
components and comment on your result.
(e) Use any other non-human image (e.g., car image, resize and crop to the same size),
and try to reconstruct it using all the PCs. Comment on your results.
(f) Rotate one of 190 individuals’ neutral expression image with different degrees and
try to reconstruct it using all PCs. Comment on your results
