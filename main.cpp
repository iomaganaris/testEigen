#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;


int main()
{
    int n;

    cout << "Size of the matrix? " << endl;
    cin >> n;

    auto m = MatrixXd::Random(n, n);
    auto v = VectorXd::Random(n);

    VectorXd s = m.partialPivLu().solve(v);

    cout << "Random matrix:\n\n";
    cout << m << endl << endl;
    cout << "Random vector:\n\n";
    cout << v << endl << endl;
    cout << "Solution (x) of M*x = v:\n\n";
    cout << s << endl << endl;

    return 0;
}