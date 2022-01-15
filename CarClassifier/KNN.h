#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
class KNN
{
private:
	size_t k;
	std::vector<std::vector<double>> X; //data matrix
	std::vector<std::string> y; //classes

	std::string predictHelper(std::vector<double> x)
	{
		//Gets k nearest samples
		std::vector<double> sortedDistances;
		std::vector<size_t> sortedIndexes;

		sortedDistances.push_back(euclideanDistance(this->X[0], x));
		sortedIndexes.push_back(0);

		size_t sizeOfData = this->X.size();

		for (size_t i = 0; i < sizeOfData; i++)
		{
			double distance = euclideanDistance(this->X[i], x);

			sortedDistances.push_back(distance);
			sortedIndexes.push_back(i);

			size_t sizeOfSorted = sortedDistances.size();

			for (size_t j = 0; j < sizeOfSorted - 1; j++)
			{
				if (sortedDistances[j] > sortedDistances[sizeOfSorted - 1])
				{
					size_t l = sizeOfSorted - 1;
					while (l != j)
					{
						sortedDistances[l] = sortedDistances[l - 1];
						sortedIndexes[l] = sortedIndexes[l - 1];
						l--;
					}

					sortedDistances[j] = distance;
					sortedIndexes[j] = i;

					break;
				}
			}
		}

		//Majority vote
		size_t max = 0;
		size_t maxIndex = 0;

		for (size_t i = 0; i < this->k; i++)
		{
			size_t counter = 0;
			for (size_t j = i; j < this->k; j++)
			{
				if (this->y[sortedIndexes[i]] == this->y[sortedIndexes[j]])
				{
					counter++;
				}
			}

			if (counter > max)
			{
				max = counter;
				maxIndex = i;
			}
		}

		return y[sortedIndexes[maxIndex]];
	}

public:
	KNN(int k)
	{
		this->k = k < 1 ? 1 : k;
	}

	static double euclideanDistance(std::vector<double> s1, std::vector<double> s2)
	{
		size_t size = s1.size();
		double sum = 0;

		for (size_t i = 0; i < size; i++)
		{
			sum += pow(s1[i] - s2[i], 2);
		}

		return sqrt(sum);
	}

	void fit(std::vector<std::vector<double>> X, std::vector<std::string> y)
	{
		this->X = X;
		this->y = y;
	}

	std::vector<std::string> predict(std::vector<std::vector<double>> X)
	{
		std::vector<std::string> result;

		if (X.size() == 1)
		{
			result.push_back(predictHelper(X[0]));
		}
		else
		{
			for (std::vector<std::vector<double>>::iterator it = X.begin(); it != X.end(); it++)
			{
				result.push_back(predictHelper(*it));
			}
		}
		return result;
	}
};