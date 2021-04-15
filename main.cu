#include "model.cuh"
#include <vector>

int main(int argc, char* argv[])
{
	try
	{
		Model nn (losses::MSE);
		nn += new Dense(10,300);
		nn += new Dense(300, 5);

		Matrix m(10);
		std::cout << nn.feed(m);

		std::vector<Matrix> train_data;
		std::vector<Matrix> ans;
		// generate some training data
		for(int i = 0; i < 10; ++i)
		{
			Matrix m(10);
			Matrix a(5, 1.0f);
			train_data.push_back(m);
			ans.push_back(a);
		}
		for (int i = 0; i < 20; ++i)
		std::cout << nn.train_batch(train_data.begin(), ans.begin(), 10, 0.01f) << "\n";
	}
	catch (char const* s)
	{
		std::cout << s << "\n";
	}
}