using Microsoft.ML.Runtime.Api;

namespace MachineLearning.Iris
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}