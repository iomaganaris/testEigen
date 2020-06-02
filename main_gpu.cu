#include <iostream>
#include <Eigen/Dense>
#include <cstdio>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

#define EIGEN_USE_GPU

__global__
void runPartialPivLuGPU(double* m, double* v, double* s, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    printf("v in device\n");
    for (int i = index; i < n; i += stride) {
        printf("%lf\n", v[i]);
    }
    typedef Eigen::Map<Eigen::MatrixXd> MapperMatrix;
    MapperMatrix m_(m, n, n);
    typedef Eigen::Map<Eigen::VectorXd> MapperVector;
    MapperVector v_(v, n);
    VectorXd s_ = m_.partialPivLu().solve(v_);
    s = s_.data();
    printf("s in device\n");
    for (int i = index; i < n; i += stride) {
        printf("%lf\n", s[i]);
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

    int n;

    cout << "Size of the matrix? " << endl;
    cin >> n;

    //MatrixXd m = MatrixXd::Random(n, n);
    //VectorXd v = VectorXd::Random(n);
    MatrixXd m(n,n);
    m << VectorXd::Zero(n), MatrixXd::Identity(n, n).block(0, 0, n, n-1);
    m = m + MatrixXd::Identity(n, n);
    VectorXd v(n);
    for (int ii = 0; ii < n-1; ++ii) {
        v[ii] = 2*ii + 1;
    }
    v[n-1] = n-1;

    VectorXd s = m.partialPivLu().solve(v);

    //MatrixXd m_device = MatrixXd::Random(n, n);
    //VectorXd v_device = VectorXd::Random(n);
    VectorXd s2 = VectorXd::Random(n);
    double* m_device = m.data();
    double* v_device = v.data();
    cout << "v_device data:" << endl;
    for(int i = 0; i < n; ++i) {
        cout << v_device[i] << endl;
    }
    double* s_device = s2.data();
    cudaMallocManaged(&m_device, n*n*sizeof(double));
    cudaMallocManaged(&v_device, n*sizeof(double));
    cudaMallocManaged(&s_device, n*sizeof(double));

    cudaError_t runPartialPivLuGPUError = cudaGetLastError();
    if(runPartialPivLuGPUError != cudaSuccess) {
        cout << "Error after MallocManaged: " << cudaGetErrorString(runPartialPivLuGPUError) << endl;
    }

    runPartialPivLuGPU<<<1, 1>>>(m_device, v_device, s_device, n);

    runPartialPivLuGPUError = cudaGetLastError();
    if(runPartialPivLuGPUError != cudaSuccess) {
        cout << "Error after kernel submission: " << cudaGetErrorString(runPartialPivLuGPUError) << endl;
    }

    cudaError_t asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) {
        cout<< "Error with cudaDeviceSync: " << cudaGetErrorString(asyncErr) << endl;
    }

    cout << "Random matrix:\n\n";
    cout << m << endl << endl;
    cout << "Random vector:\n\n";
    cout << v << endl << endl;
    cout << "Solution (x) of M*x = v:\n\n";
    cout << s << endl << endl;
    cout << "Device Solution (x) of M*x = v:\n\n";
    cout << s_device << endl;

    cudaFree((void*)&m);
    cudaFree((void*)&v);
    cudaFree((void*)&s_device);

    return 0;
}
