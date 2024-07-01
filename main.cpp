#include <iostream>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class LinearRegression {
private:
    double sumX;
    double sumY;
    double sumXY;
    double sumX2;
    double sumY2;

    void calculateSummations(const std::vector<double>& x, const std::vector<double>& y) {
        sumX = 0;
        sumY = 0;
        sumXY = 0;
        sumX2 = 0;
        sumY2 = 0;

        for (size_t i = 0; i < x.size(); ++i) {
            double xi = x[i];
            double yi = y[i];

            sumX += xi;
            sumY += yi;
            sumXY += xi * yi;
            sumX2 += xi * xi;
            sumY2 += yi * yi;
        }
    }

public:
    LinearRegression() : sumX(0), sumY(0), sumXY(0), sumX2(0), sumY2(0) {}

    double calculateM(const std::vector<double>& x, const std::vector<double>& y) {
        calculateSummations(x, y);
        int n = x.size();

        double numerator = (n * sumXY) - (sumX * sumY);
        double denominator = (n * sumX2) - (sumX * sumX);

        return numerator / denominator;
    }

    double calculateB(const std::vector<double>& x, const std::vector<double>& y) {
        calculateSummations(x, y);
        int n = x.size();

        double numerator = (sumY * sumX2) - (sumX * sumXY);
        double denominator = (n * sumX2) - (sumX * sumX);

        return numerator / denominator;
    }

    double calculateR(const std::vector<double>& x, const std::vector<double>& y) {
        calculateSummations(x, y);
        int n = x.size();

        double numerator = (n * sumXY) - (sumX * sumY);
        double denominator = std::sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        return numerator / denominator;
    }
};

void loadCSV(const std::string& filename, std::vector<double>& col1, std::vector<double>& col2, int col1Index, int col2Index) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int colIndex = 0;
        double val1, val2;
        bool val1Set = false, val2Set = false;
        bool inQuotes = false;
        char c;

        value.clear();
        while (ss.get(c)) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                // When we encounter a comma and we are not inside quotes, we have a complete value
                if (colIndex == col1Index) {
                 
                    try {
                        val1 = std::stod(value);
                        val1Set = true;
                    } catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid number in line: " << line << std::endl;
                    } catch (const std::out_of_range& e) {
                        std::cerr << "Number out of range in line: " << line << std::endl;
                    }
                }
                if (colIndex == col2Index) {
                
                    try {
                        val2 = std::stod(value);
                        val2Set = true;
                    } catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid number in line: " << line << std::endl;
                    } catch (const std::out_of_range& e) {
                        std::cerr << "Number out of range in line: " << line << std::endl;
                    }
                }
                value.clear();
                ++colIndex;
            } else {
                value += c;
            }
        }

        // Handle the last value after the loop
        if (!value.empty()) {
            if (colIndex == col1Index) {
                
                try {
                    val1 = std::stod(value);
                    val1Set = true;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number in line: " << line << std::endl;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Number out of range in line: " << line << std::endl;
                }
            }
            if (colIndex == col2Index) {
               
                try {
                    val2 = std::stod(value);
                    val2Set = true;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number in line: " << line << std::endl;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Number out of range in line: " << line << std::endl;
                }
            }
        }

        if (val1Set && val2Set) {
            col1.push_back(val1);
            col2.push_back(val2);
        }
    }

    file.close();
}


void filterCSV(const std::vector<double>& col1, const std::vector<double>& col2, std::vector<double>& filteredCol1, std::vector<double>& filteredCol2, double col1Max, double col2Max) {
    for (size_t i = 0; i < col1.size(); ++i) {
        if (col1[i] <= col1Max && col2[i] <= col2Max) {
            filteredCol1.push_back(col1[i]);
            filteredCol2.push_back(col2[i]);
        }
    }
}


double predictResult(double x, double m, double b)
{
    return (m*x + b);
}
int main() {
    

    std::vector<double> x;
    std::vector<double> y;
    std::string filename = "articulos_ml.csv";

    try {
        loadCSV(filename, x, y, 2, 7); // Aquí se toman las columnas 0 y 1, puedes cambiarlo según sea necesario
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    

    std::vector<double> filteredX;
    std::vector<double> filteredY;
    filterCSV(x, y, filteredX, filteredY, 3500, 80000);

    // Guardar datos filtrados en un archivo CSV
    std::ofstream outputFile("datos_filtrados.csv");
    if (outputFile.is_open()) {
        for (size_t i = 0; i < filteredX.size(); ++i) {
            outputFile << filteredX[i] << "," << filteredY[i] << "\n";
        }
        outputFile.close();
        std::cout << "Datos filtrados guardados en datos_filtrados.csv\n";
    } else {
        std::cerr << "Error al abrir el archivo para escritura\n";
        return 1;
    }
    

    LinearRegression lr;
    double m = lr.calculateM(filteredX, filteredY);
    double b = lr.calculateB(filteredX, filteredY);
    double r = lr.calculateR(filteredX, filteredY);

    std::cout << "Pendiente (m): " << m << std::endl;
    std::cout << "Ordenada al origen (b): " << b << std::endl;
    std::cout << "Coeficiente de correlacion (r): " << r << std::endl;
    std::cout << "Ecuacion (y): " << m << "x + " << b << std::endl;

    std::cout << "\nPredecir Resultado: " << "x =  2000 " << std::endl;
    std::cout << "Ecuacion (y): " << m << " * 2000 + " << b << std::endl;
    double re =  predictResult(2000,m,b);
    std::cout << "Resultado (y): " << re << std::endl;
    

    return 0;
}
