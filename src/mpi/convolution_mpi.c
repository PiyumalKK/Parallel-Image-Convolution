#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "../../include/image_utils.h"

float* generate_gaussian_kernel(int size, float sigma) {

    float *kernel = malloc(size * size * sizeof(float));
    int half = size / 2;
    float sum = 0;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {

            float value = expf(-(x*x + y*y) / (2*sigma*sigma));

            kernel[(y+half)*size + (x+half)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < size*size; i++)
        kernel[i] /= sum;

    return kernel;
}

float edge_detection[9] = {
    -1,-1,-1,
    -1, 8,-1,
    -1,-1,-1
};

float sharpen[9] = {
     0,-1,0,
    -1,5,-1,
     0,-1,0
};

unsigned char apply_kernel(Image *img, int x, int y, int c, float *kernel, int ksize) {

    int half = ksize/2;
    float sum = 0;

    for(int ky=-half; ky<=half; ky++){
        for(int kx=-half; kx<=half; kx++){

            int ix=x+kx;
            int iy=y+ky;

            if(ix<0) ix=0;
            if(iy<0) iy=0;
            if(ix>=img->width) ix=img->width-1;
            if(iy>=img->height) iy=img->height-1;

            int img_index=(iy*img->width+ix)*img->channels+c;
            int kernel_index=(ky+half)*ksize+(kx+half);

            sum+=img->data[img_index]*kernel[kernel_index];
        }
    }

    if(sum<0) sum=0;
    if(sum>255) sum=255;

    return (unsigned char)sum;
}

int main(int argc,char *argv[]) {

    MPI_Init(&argc,&argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank==0){

        Image *input=load_image(argv[1]);
        Image *output=create_image(input->width,input->height,input->channels);

        float *kernel;
        int kernel_size;

        if(strcmp(argv[3],"blur")==0){
            kernel_size=21;
            kernel=generate_gaussian_kernel(kernel_size,7.0);
        }
        else if(strcmp(argv[3],"edge")==0){
            kernel=edge_detection;
            kernel_size=3;
        }
        else{
            kernel=sharpen;
            kernel_size=3;
        }

        for(int y=0;y<input->height;y++){
            for(int x=0;x<input->width;x++){
                for(int c=0;c<input->channels;c++){

                    int index=(y*input->width+x)*input->channels+c;

                    output->data[index]=
                        apply_kernel(input,x,y,c,kernel,kernel_size);
                }
            }
        }

        save_image(argv[2],output);

        free_image(input);
        free_image(output);
    }

    MPI_Finalize();
}