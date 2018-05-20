using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearning.Iris.Controllers
{
    [Route("api/irises")]
    public class IrisesController : Controller
    {
        [HttpPost]
        public IActionResult Predict([FromBody]Iris iris)
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<IrisData>("iris-data.txt", separator: ","),
                new Dictionarizer("Label"),
                new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"),
                new StochasticDualCoordinateAscentClassifier(),
                new PredictedLabelColumnOriginalValueConverter {PredictedLabelColumn = "PredictedLabel"}
            };

            var data = new IrisData
            {
                PetalLength = iris.PetalLength,
                PetalWidth = iris.PetalWidth,
                SepalLength = iris.SepalLength,
                SepalWidth = iris.SepalWidth
            };

            var model = pipeline.Train<IrisData, IrisPrediction>();
            var prediction = model.Predict(data);

            return Ok(prediction);
        }
    }
}
