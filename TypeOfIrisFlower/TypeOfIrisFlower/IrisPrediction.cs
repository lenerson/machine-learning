using Microsoft.ML.Runtime.Api;

namespace TypeOfIrisFlower
{
    public sealed class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
