import inputParser
import k_calculator
import k_means

if __name__ == '__main__':
    print('Input dataset, formats accepted: \n'
          '1 .csv file: point1x,point1y\n'
          '2. user input in format: point1x,point1y point2x,point2y\n'
          '3. random generation: input a dataset size integer n\n'
          'Please, input your dataset: ')
    rawInput = input()

    dataset = inputParser.parse(rawInput)

    print("Your dataset contains: %d datapoints" %len(dataset),end="\n\n")

    k_calculator.silhouetteScores(dataset)
    best_k = k_means.calculate_k(dataset)
    centroids = k_means.genereateCentroids(best_k,dataset)
    k_means.plotGraph(centroids)




