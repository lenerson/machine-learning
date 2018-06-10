using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace TypeOfIrisFlower
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create a pipeline and load data
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader("iris.data.txt").CreateFrom<IrisData>(separator: ','));

            // Transform data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (What type of iris is this?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Convert the Label back into original text (after converting to number)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // Train model based on the data set
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // Use model to make a prediction
            // Change these numbers to test different predictions
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.ReadKey();
        }
    }
}
