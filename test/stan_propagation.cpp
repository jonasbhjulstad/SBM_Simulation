//demonstrate integration with stan
#include <stan/

int main()
{
    using namespace stan::math;
    using namespace std;

    //define variables
    var x = 0.0;
    var y = 0.0;
    var z = 0.0;

    //define function
    var f = x*x + y*y + z*z;

    //define gradient
    vector<var> grad;
    grad.push_back(x);
    grad.push_back(y);
    grad.push_back(z);

    //compute gradient
    vector<double> g = gradient(f, grad);

    //print gradient
    cout << "Gradient: " << g[0] << " " << g[1] << " " << g[2] << endl;

    return 0;
}
