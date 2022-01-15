#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include "KNN.h"
#include <chrono>

double PERCENTAGE_FOR_TRAINING = 0.7;
size_t K = 3;

//already a number
const size_t MODEL_POSITION = 0;
const size_t YEAR_POSITION = 1;
const size_t PRICE_POSITION = 2;
const size_t MILEAGE_POSITION = 4;
const size_t TAX_POSITION = 6;
const size_t MPG_POSITION = 7;
const size_t ENGINE_SIZE_POSITION = 8;

//numbers after new columns
const size_t PROCESSED_YEAR_POSITION = YEAR_POSITION - 1;
const size_t PROCESSED_PRICE_POSITION = PRICE_POSITION - 1;
const size_t PROCESSED_MILEAGE_POSITION = MILEAGE_POSITION + 1;
const size_t PROCESSED_TAX_POSITION = TAX_POSITION + 4;
const size_t PROCESSED_MPG_POSITION = MPG_POSITION + 4;
const size_t PROCESSED_ENGINE_SIZE_POSITION = ENGINE_SIZE_POSITION + 4;

//needs to be processed to a number
const size_t TRANSMISSION_POSITION = 3;

//functions
//splits data into parts for training and parts for testing
void trainTestSplit(std::vector<std::vector<double>>& X, std::vector<std::string>& y,
    std::vector<std::vector<double>>& XTrain, std::vector<std::vector<double>>& XTest,
    std::vector<std::string>& yTrain, std::vector<std::string>& yTest);

//normalizes columns (divides each element of the vector by the lenght of the vector this way the new vector has length = 1)
void normalize(std::vector<std::vector<double>>& values, size_t col);

//collects the raw data from the file
std::vector<std::vector<std::string>> readFromFile(std::ifstream& file);

//separates the classes names from the raw data
std::vector<std::string> getCLasses(std::vector<std::vector<std::string>> rawData);

//pre-processes the raw data so it can be used for kNN
std::vector<std::vector<double>> preProcessData(std::vector<std::vector<std::string>> rawData);

//input brand
std::string brandChoiceMenu();

//maNuAl -> MANUAL
void allToCaps(std::string& str);

//custom sample input
std::vector<std::vector<double>> enterSample();

//choose to find out a custom sample's model or test the knn algorithm accuracy
bool chooseMode();

int main()
{
    //choose the car brand
    std::string brandFileName = brandChoiceMenu();

    //choose to test the knn accuracy or use the algorithm to find out a custom sample's model
    bool isInTestMode = chooseMode();

    auto frame_start = std::chrono::high_resolution_clock::now(); //starts a timer

    //open csv file containing the data set
    std::ifstream dataSetFile(brandFileName);

    if (!dataSetFile.is_open())
    {
        std::cerr << "File did not open.\n";
        return 0;
    }

    //get the data and pre-process it
    std::vector<std::vector<std::string>> rawData = readFromFile(dataSetFile);
    dataSetFile.close(); //close the file as we no longer need it

    std::vector<std::string> classes = getCLasses(rawData);

    std::vector<std::vector<double>> data = preProcessData(rawData);

    //normalize the data
    normalize(data, PROCESSED_YEAR_POSITION);
    normalize(data, PROCESSED_PRICE_POSITION);
    normalize(data, PROCESSED_MILEAGE_POSITION);
    normalize(data, PROCESSED_TAX_POSITION);
    normalize(data, PROCESSED_MPG_POSITION);
    normalize(data, PROCESSED_ENGINE_SIZE_POSITION);

    //split the data into parts for testing and parts for training
    std::vector<std::vector<double>> XTrain;
    std::vector<std::vector<double>> XTest;

    std::vector<std::string> yTrain;
    std::vector<std::string> yTest;

    trainTestSplit(data, classes, XTrain, XTest, yTrain, yTest);

    //start kNN
    KNN model(K);
    model.fit(XTrain, yTrain);

    if (isInTestMode)
    {
        std::cout << "\nPlease wait a few second...\n\n";
        std::vector<std::string> yPred = model.predict(XTest);

        size_t correct = 0;
        size_t sizeYPred = yPred.size();

        for (size_t i = 0; i < sizeYPred; i++)
        {
            if (yPred[i] == yTest[i])
            {
                correct++;
            }
        }

        std::cout << correct << " out of " << sizeYPred << " are correct\n";
        std::cout << "Success rate: " << (double(correct) / double(sizeYPred)) * 100 << "%\n";

        auto frame_end = std::chrono::high_resolution_clock::now(); //end of timer

        std::chrono::duration<float> duration = frame_end - frame_start;
        std::cout << "Run time: " << duration.count() << " seconds\n\n";
    }
    else
    {
        std::cout << "\nCar model is: " << model.predict(enterSample())[0] << "\n\n";
    }

    //Restart the program or end it
    std::cout << "Choose:\n"
        << "1. exit\n"
        << "2. run\n"
        << "Enter the number of your choise\n";

    int exitOrRun = 0;

    while (exitOrRun < 1 || exitOrRun > 2)
    {
        std::cin >> exitOrRun;
        if (exitOrRun < 1 || exitOrRun > 2)
            std::cout << "You must enter a number between 1 and 2 corresponding to your choice!\n";
    }

    if (exitOrRun == 2)
    {
        main();
    }
}

void trainTestSplit(std::vector<std::vector<double>>& X, std::vector<std::string>& y,
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
    for (size_t i = 0; i < size; i++)
    {
        sum += pow(values[i][col], 2);
    }

    double magnitude = sqrt(sum);

    for (size_t i = 0; i < size; i++)
    {
        values[i][col] /= magnitude;
    }
}

std::vector<std::vector<std::string>> readFromFile(std::ifstream& file)
{
    std::string str = "";
    std::vector<std::vector<std::string>> rawData; //the data we have before we process it
    std::vector<std::string> line; //used to separate each sample
    bool isFirstLine = true;
    char symbol = '\0';

    while (symbol != -1) //savess the data from the file to rawData
    {
        symbol = file.get();

        if (isFirstLine)
        {
            if (symbol == ',')
            {
                str = "";
                continue;
            }
            else if (symbol == '\n')
            {
                isFirstLine = false;
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

    return rawData;
}

std::vector<std::string> getCLasses(std::vector<std::vector<std::string>> rawData)
{
    std::vector<std::string> classes; //y or the model of the car

    size_t rowsRawData;
    rowsRawData = rawData.size();

    for (size_t i = 0; i < rowsRawData; i++) //separates classes from the rest of the data
    {
        classes.push_back(rawData[i][MODEL_POSITION]);
    }

    return classes;
}

std::vector<std::vector<double>> preProcessData(std::vector<std::vector<std::string>> rawData)
{
    std::vector<std::vector<double>> data; //the procesed data
    std::vector<double> sample;

    size_t rowsRawData, colsRawData;
    rowsRawData = rawData.size();
    colsRawData = rawData[0].size();

    double num = 0;

    double numberValue = 0;
    for (size_t i = 0; i < rowsRawData; i++) //processes and saves all the samples in an array
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

    return data;
}

std::string brandChoiceMenu()
{
    std::cout << "Choose a brand:\n"
        << "1. toyota\n"
        << "2. ford\n"
        << "3. volkswagen\n"
        << "4. skoda\n"
        << "5. hyundi\n"
        << "Enter the number of the brand you want choose\n";

    int brandChoice = 0;

    while (brandChoice < 1 || brandChoice > 5)
    {
        std::cin >> brandChoice;
        if (brandChoice < 1 || brandChoice > 5)
            std::cout << "You must enter a number between 1 and 5 corresponding to the brand you want to choose!\n";
    }

    switch (brandChoice)
    {
    case 1:
        return "toyota.csv";
    case 2:
        return "ford.csv";
    case 3:
        return "volkswagen.csv";
    case 4:
        return "skoda.csv";
    case 5:
        return "hyundi.csv";
    default:
        return "toyota.csv";
    }

}

void allToCaps(std::string& str)
{
    for (std::string::iterator it = str.begin(); it != str.end(); it++)
    {
        if (*it <= 'z' && *it >= 'a')
            *it += 'A' - 'a';
    }
}

std::vector<std::vector<double>> enterSample()
{
    std::vector<std::vector<double>> result;
    std::vector<double> sample;
    double value;
    std::string textValue;

    //year
    std::cout << "Year = ";
    std::cin >> value;
    sample.push_back(value);

    //price
    std::cout << "Price = ";
    std::cin >> value;
    sample.push_back(value);

    //transmission
    std::cout << "Transmission type: ";
    std::cin >> textValue;
    allToCaps(textValue);

    if (textValue == "MANUAL")
    {
        sample.push_back(1);
        sample.push_back(0);
        sample.push_back(0);
    }
    else if (textValue == "AUTOMATIC")
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

    //mileage
    std::cout << "Mileage = ";
    std::cin >> value;
    sample.push_back(value);

    //fuelType
    std::cout << "Fuel type: ";
    std::cin >> textValue;
    allToCaps(textValue);

    if (textValue == "Petrol")
    {
        sample.push_back(1);
        sample.push_back(0);
        sample.push_back(0);
        sample.push_back(0);
    }
    else if (textValue == "Diesel")
    {
        sample.push_back(0);
        sample.push_back(1);
        sample.push_back(0);
        sample.push_back(0);
    }
    else if (textValue == "Hybrid")
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

    //tax
    std::cout << "Tax = ";
    std::cin >> value;
    sample.push_back(value);

    //mpg
    std::cout << "Miles-per-gallon = ";
    std::cin >> value;
    sample.push_back(value);

    //engineSize
    std::cout << "Engine size: ";
    std::cin >> value;
    sample.push_back(value);

    result.push_back(sample);

    normalize(result, PROCESSED_YEAR_POSITION);
    normalize(result, PROCESSED_PRICE_POSITION);
    normalize(result, PROCESSED_MILEAGE_POSITION);
    normalize(result, PROCESSED_TAX_POSITION);
    normalize(result, PROCESSED_MPG_POSITION);
    normalize(result, PROCESSED_ENGINE_SIZE_POSITION);

    return result;
}

bool chooseMode()
{
    std::cout << "Choose mode:\n"
        << "1. test algorithm accuracy\n"
        << "2. find car model\n"
        << "Enter the number of the brand you want choose\n";

    int modeChoice = 0;

    while (modeChoice < 1 || modeChoice > 2)
    {
        std::cin >> modeChoice;
        if (modeChoice < 1 || modeChoice > 2)
            std::cout << "You must enter a number between 1 and 2 corresponding to the mode you want to choose!\n";
    }

    switch (modeChoice)
    {
    case 1:
        return true;
    case 2:
        return false;
    }
}