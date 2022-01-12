#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include "KNN.h"
#include <chrono>

double PERCENTAGE_FOR_TRAINING = 0.7; //by testing we decided that 80% and k=5 is the best combination and it is used when we input custom patients
size_t K = 3;

static void trainTestSplit(std::vector<std::vector<double>>& X, std::vector<std::string>& y,
    std::vector<std::vector<double>>& XTrain, std::vector<std::vector<double>>& XTest,
    std::vector<std::string>& yTrain, std::vector<std::string>& yTest)
{
    srand((unsigned)time(NULL));

    size_t size = X.size();
    size_t limit = (size_t)(size * PERCENTAGE_FOR_TRAINING);
    ;
    for (size_t i = 0; i < size; i++) //randomizes the order of the samples from the data
    {
        size_t rndIndex = rand() % size;
        std::swap(X[i], X[rndIndex]);
        std::swap(y[i], y[rndIndex]);
    }

    //separates samples for training
    for (size_t i = 0; i < limit; i++)
    {
        XTrain.push_back(X[i]);
        yTrain.push_back(y[i]);
    }

    //separates samples for testing
    for (size_t i = limit; i < size; i++)
    {
        XTest.push_back(X[i]);
        yTest.push_back(y[i]);
    }
}

void normalize(std::vector<std::vector<double>>& values, size_t col)
{
    double sum = 0;
    size_t size = values.size();
    for (size_t i = 0; i < size; i += 10)
    {
        sum += pow(values[i][col], 2);
    }

    double magnitude = sqrt(sum);

    for (size_t i = 0; i < size; i += 10)
    {
        values[i][col] /= magnitude;
    }
}

int main()
{
    auto frame_start = std::chrono::high_resolution_clock::now();

    std::ifstream file("toyota.csv");

    if (!file.is_open())
    {
        std::cerr << "File did not open.\n";
        return 0;
    }

    //already a number
    const size_t MODEL_POSITION = 0;
    const size_t YEAR_POSITION = 1;
    const size_t PRICE_POSITION = 2;
    const size_t MILEAGE_POSITION = 4;
    const size_t TAX_POSITION = 6;
    const size_t MPG_POSITION = 7;
    const size_t ENGINE_SIZE_POSITION = 8;

    //needs to be processed to a number
    const size_t TRANSMISSION_POSITION = 3;




    std::string str = "";
    std::vector<std::vector<std::string>> rawData; //the data we have before we process it
    std::vector<std::string> line; //used to separate each sample
    std::vector<std::string> labels; //used to keep the name of each column
    bool isFirstLine = true;
    char symbol = '\0';

    std::cout << "Started loading data\n";
    while (symbol != -1) //savess the data from the file to rawData
    {
        symbol = file.get();

        if (isFirstLine)
        {
            if (symbol == ',')
            {
                labels.push_back(str);
                str = "";
                continue;
            }
            else if (symbol == '\n')
            {
                isFirstLine = false;
                labels.push_back(str);
                str = "";
                continue;
            }
        }
        else
        {
            if (symbol == ',')
            {
                line.push_back(str);
                str = "";
                continue;
            }
            else if (symbol == '\n')
            {
                line.push_back(str);
                str = "";
                rawData.push_back(line);
                while (!line.empty())
                {
                    line.pop_back();
                }
                continue;
            }
        }

        str += symbol;
    }
    std::cout << "Finished loading data\n";

    std::vector<std::vector<double>> data; //the procesed data
    std::vector<double> sample;
    std::vector<std::string> classes; //y or the types of drugs

    size_t rowsRawData, colsRawData;
    rowsRawData = rawData.size();
    colsRawData = rawData[0].size();

    double num = 0;

    std::cout << "Loading classes\n";
    for (size_t i = 0; i < rowsRawData; i += 10) //separates classes from the rest of the data
    {
        classes.push_back(rawData[i][MODEL_POSITION]);
    }
    std::cout << "Finished loading classes\n";

    std::cout << "Processing data\n";
    double numberValue = 0;
    for (size_t i = 0; i < rowsRawData; i += 10) //processes and saves all the samples in an array
    {
        for (size_t j = 1; j < colsRawData; j++)
        {
            if (j == YEAR_POSITION || j == PRICE_POSITION ||
                j == MILEAGE_POSITION || j == TAX_POSITION || j == MPG_POSITION ||
                j == ENGINE_SIZE_POSITION)
            {
                sample.push_back(std::atof(rawData[i][j].c_str()));
            }
            else if (j == TRANSMISSION_POSITION)
            {
                if (rawData[i][j] == "Manual")
                {
                    sample.push_back(1);
                    sample.push_back(0);
                    sample.push_back(0);
                }
                else if (rawData[i][j] == "Automatic")
                {
                    sample.push_back(0);
                    sample.push_back(1);
                    sample.push_back(0);
                }
                else //if Semi-Auto
                {
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(1);
                }
            }
            else //fuelType column
            {
                if (rawData[i][j] == "Petrol")
                {
                    sample.push_back(1);
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(0);
                }
                else if (rawData[i][j] == "Diesel")
                {
                    sample.push_back(0);
                    sample.push_back(1);
                    sample.push_back(0);
                    sample.push_back(0);
                }
                else if (rawData[i][j] == "Hybrid")
                {
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(1);
                    sample.push_back(0);
                }
                else //Other
                {
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(1);
                }
            }
        }

        data.push_back(sample);
        while (!sample.empty())
        {
            sample.pop_back();
        }
    }
    std::cout << "Finished processing data\n";

    //Normalize data
    const size_t PROCESSED_YEAR_POSITION = YEAR_POSITION - 1;
    const size_t PROCESSED_PRICE_POSITION = PRICE_POSITION - 1;
    const size_t PROCESSED_MILEAGE_POSITION = MILEAGE_POSITION + 1;
    const size_t PROCESSED_TAX_POSITION = TAX_POSITION + 4;
    const size_t PROCESSED_MPG_POSITION = MPG_POSITION + 4;
    const size_t PROCESSED_ENGINE_SIZE_POSITION = ENGINE_SIZE_POSITION + 4;

    //normalize(data, PROCESSED_YEAR_POSITION);
    normalize(data, PROCESSED_PRICE_POSITION);  
    normalize(data, PROCESSED_MILEAGE_POSITION);
    normalize(data, PROCESSED_TAX_POSITION); //
    //normalize(data, PROCESSED_MPG_POSITION);
    //normalize(data, PROCESSED_ENGINE_SIZE_POSITION);
    

    size_t rowsData, colsData;
    rowsData = data.size();
    colsData = data[0].size();

    ////year
    double sum = 0;
    size_t counter = 0;
    for (size_t i = 0; i < rowsData; i += 10)
    {
        counter++;
        sum += data[i][PROCESSED_YEAR_POSITION];
    }

    //double avg = sum / counter;

    //for (size_t i = 0; i < rowsData; i += 10)
    //{
    //    data[i][PROCESSED_YEAR_POSITION] -= avg;
    //    data[i][PROCESSED_YEAR_POSITION] /= 10;
    //}

    //tax
    //std::cout << "TAX:" << data[0][PROCESSED_TAX_POSITION] << "\n";
    //for (size_t i = 0; i < rowsData; i += 10)
    //{
    //    data[i][PROCESSED_TAX_POSITION] /= 50;
    //}

    //////mpg
    std::cout << "MPG:" << data[0][PROCESSED_MPG_POSITION] << "\n";
    for (size_t i = 0; i < rowsData; i += 10)
    {
        data[i][PROCESSED_MPG_POSITION] /= 50;
    }

    std::vector<std::vector<double>> XTrain;
    std::vector<std::vector<double>> XTest;

    std::vector<std::string> yTrain;
    std::vector<std::string> yTest;

    std::cout << "Started spliting data\n";
    trainTestSplit(data, classes, XTrain, XTest, yTrain, yTest);
    std::cout << "Finished spliting data\n";

    KNN model(K);

    std::cout << "Started fit\n";
    model.fit(XTrain, yTrain);
    std::cout << "Finisjed fit\n";
    std::cout << "Started predicting\n";
    std::vector<std::string> yPred = model.predict(XTest);

    size_t correct = 0;
    size_t size = yPred.size();

   
    for (size_t i = 0; i < size; i++)
    {
        if (yPred[i] == yTest[i])
        {
            correct++;
        }
    }

    std::cout << correct << " out of " << size << " are correct\n";
    std::cout << "Success rate: " << (double(correct) / double(size)) * 100 << "%\n";

    file.close();


    auto frame_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration = frame_end - frame_start;
    std::cout << duration.count() << '\n';
}
