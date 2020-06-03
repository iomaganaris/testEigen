#include <iostream>
#include <Eigen/Dense>
#include <cstdio>

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;
using Eigen::Vector4d;
using namespace std;

//#define EIGEN_USE_GPU

__global__
void runPartialPivLuGPU(Matrix<double, 4, 4>& m, Vector4d& v, Vector4d& s, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    printf("v in device\n");
    for (int i = index; i < n; i += stride) {
        printf("v[%d] = %lf\n", i, v(i));
    }
    /*typedef Eigen::Map<Eigen::MatrixXd> MapperMatrix;
    MapperMatrix m_(m, n, n);
    typedef Eigen::Map<Eigen::VectorXd> MapperVector;
    MapperVector v_(v, n);*/
    s = m.partialPivLu().solve(v);
    //s = s_.data();
    printf("s in device\n");
    for (int i = index; i < n; i += stride) {
        printf("s[%d] = %lf\n", i, s(i));
    }
}

int main()
{

    // GPU initializations
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId); 

    int threadsPerBlock = 256;
    int numberOfBlocks = 32 * numberOfSMs;

    const int n = 4;

    //cout << "Size of the matrix? " << endl;
    //cin >> n;

    //MatrixXd m = MatrixXd::Random(n, n);
    //VectorXd v = VectorXd::Random(n);
    Matrix<double, 4, 4> *m;
    cudaMallocManaged((void**)&m, sizeof(Matrix<double, 4, 4>));
    *m << VectorXd::Zero(n), MatrixXd::Identity(n, n).block(0, 0, n, n-1);
    *m = *m + MatrixXd::Identity(n, n);
    Vector4d *v;
    cudaMallocManaged((void**)&v, sizeof(Vector4d));
    for (int ii = 0; ii < n-1; ++ii) {
        (*v)[ii] = 2*ii + 1;
    }
    (*v)[n-1] = n-1;

    Vector4d s = (*m).partialPivLu().solve(*v);

    Vector4d *s_device;
    cudaMallocManaged((void**)&s_device, sizeof(Vector4d));

    cudaError_t runPartialPivLuGPUError = cudaGetLastError();
    if(runPartialPivLuGPUError != cudaSuccess) {
        cout << "Error after MallocManaged: " << cudaGetErrorString(runPartialPivLuGPUError) << endl;
    }

    runPartialPivLuGPU<<<1, 1>>>(*m, *v, *s_device, n);

    runPartialPivLuGPUError = cudaGetLastError();
    if(runPartialPivLuGPUError != cudaSuccess) {
        cout << "Error after kernel submission: " << cudaGetErrorString(runPartialPivLuGPUError) << endl;
    }

    cudaError_t asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) {
        cout<< "Error with cudaDeviceSync: " << cudaGetErrorString(asyncErr) << endl;
    }

    cout << "Random matrix:\n\n";
    cout << *m << endl << endl;
    cout << "Random vector:\n\n";
    cout << *v << endl << endl;
    cout << "Solution (x) of M*x = v:\n\n";
    cout << s << endl << endl;
    cout << "Device Solution (x) of M*x = v:\n\n";
    cout << *s_device << endl;

    cudaFree((void*)&m);
    cudaFree((void*)&v);
    cudaFree((void*)&s_device);

    return 0;
}
