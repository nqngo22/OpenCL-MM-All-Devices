/* OpenCL Matrix multiplication using all devices */

#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "/matvec.cl"
#define KERNEL_FUNC "matvec_mult"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <OpenCL/cl.h>


int main() {
    cl_platform_id *platforms;
    cl_device_id *devices;
    cl_context context;
    cl_command_queue *queues;
    cl_uint num_platforms;
    cl_int i, err;
    
    
    
    
    /* Program/kernel data structures */
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_kernel kernel;
    
    /* Data and buffers */
    float mat[16], vec[4], result[4];
    float correct[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cl_mem mat_buff, vec_buff, res_buff;
    size_t work_units_per_kernel;
    
    /* Initialize data to be processed by the kernel */
    for(i=0; i<16; i++) {
        mat[i] = i * 2.0f;
    }
    for(i=0; i<4; i++) {
        vec[i] = i * 3.0f;
        correct[0] += mat[i]    * vec[i];
        correct[1] += mat[i+4]  * vec[i];
        correct[2] += mat[i+8]  * vec[i];
        correct[3] += mat[i+12] * vec[i];
    }
    
    
    
    
    /* Identify platforms */
    err = clGetPlatformIDs(5, NULL, &num_platforms);
    if(err < 0){
        perror("Couldn't find any platforms");
        exit(1);
    }
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
    printf("num platforms: %i\n", num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    /* Identify devices */
    cl_uint num_devices;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 5, NULL, &num_devices);
    devices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 5, devices, &num_devices);
    
    for (int i=0; i<num_devices; i++)
    {
        char buffer[10240];
        printf("  -- %d --\n", i);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_NAME = %s\n", buffer);
    }
    
    queues= (cl_command_queue*) malloc(sizeof(cl_command_queue) * num_devices);
    
    /* Create the context */
    context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
    if(err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }
    
    /* Read program file and place content into buffer */
    //Determine size of source file
    program_handle = fopen(PROGRAM_FILE, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    //Read file content into buffer
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    
    /* Create program from file/buffer */
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);
    
    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        perror("clBuildProgram failed");
        exit(1);
    }
    
    /* Create kernel for the mat_vec_mult function */
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if(err < 0) {
        perror("Couldn't create the kernel");
        exit(1);
    }
    
    /* Create CL buffers to hold input and output data */
    mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
                              CL_MEM_COPY_HOST_PTR, sizeof(float)*16, mat, &err);
    if(err < 0) {
        perror("Couldn't create a buffer object");
        exit(1);
    }
    vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
                              CL_MEM_COPY_HOST_PTR, sizeof(float)*4, vec, NULL);
    res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                              sizeof(float)*4, NULL, NULL);
    
    /* Create kernel arguments from the CL buffers */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
    if(err < 0) {
        perror("Couldn't set the kernel argument");
        exit(1);
    }
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);
    for (i = 0 ; i < num_devices; i++){
        /* Create a CL command queue for the device*/
        queues[i] = clCreateCommandQueue(context, devices[i], 0, &err);
        if(err < 0) {
            perror("Couldn't create the command queue");
            exit(1);
        }
        
        work_units_per_kernel = 4; /* 4 work-units per kernel */
        err = clEnqueueNDRangeKernel(queues[i], kernel, 1, NULL, &work_units_per_kernel,
                                     NULL, 0, NULL, NULL);
        if(err < 0) {
            perror("Couldn't enqueue the kernel execution command");
            exit(1);
        }
        
        /* Read the result */
        err = clEnqueueReadBuffer(queues[i], res_buff, CL_TRUE, 0, sizeof(float)*4,
                                  result, 0, NULL, NULL);
        if(err < 0) {
            perror("Couldn't enqueue the read buffer command");
            exit(1);
        }
    }
    
    /* Test the result */
    if((result[0] == correct[0]) && (result[1] == correct[1])
       && (result[2] == correct[2]) && (result[3] == correct[3])) {
        printf("Matrix-vector multiplication successful.\n");
    }
    else {
        printf("Matrix-vector multiplication unsuccessful.\n");
    }
    
    /* Deallocate resources */
    clReleaseMemObject(mat_buff);
    clReleaseMemObject(vec_buff);
    clReleaseMemObject(res_buff);
    clReleaseKernel(kernel);
    for(i = 0; i < num_devices; i++){
        clReleaseCommandQueue(queues[i]);
    }
    clReleaseProgram(program);
    clReleaseContext(context);
    
    
    return 0;
}
