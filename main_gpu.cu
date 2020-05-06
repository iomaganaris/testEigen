#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;


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

    MatrixXd m = MatrixXd::Random(n, n);
    VectorXd v = VectorXd::Random(n);

    VectorXd s = m.partialPivLu().solve(v);

    cout << "Random matrix:\n\n";
    cout << m << endl << endl;
    cout << "Random vector:\n\n";
    cout << v << endl << endl;
    cout << "Solution (x) of M*x = v:\n\n";
    cout << s << endl << endl;

    return 0;
}
