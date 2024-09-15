using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using StockPredictionAPI.Models; 

[Route("api/[controller]")]
[ApiController]
public class StockPredictionController : ControllerBase
{
    private const string apiKey = "fdBhj6FJhbAKaYyTh5fU3pwUvPY5X32E";
    private readonly HttpClient _httpClient;
    private readonly MLContext mlContext;

    public StockPredictionController()
    {
        _httpClient = new HttpClient();
        mlContext = new MLContext();
    }

    [HttpGet("{symbol}")]
    public async Task<IActionResult> GetPrediction(string symbol)
    {
         // Load historical data
        var historicalData = await LoadHistoricalData(symbol);
        if (historicalData == null || historicalData.Historical.Count == 0)
        {
            return BadRequest("No historical data available.");
        }

        // Convert historical data to IDataView
        var data = mlContext.Data.LoadFromEnumerable(historicalData.Historical);

        // Define data preparation and model training pipeline
        var pipeline = mlContext.Transforms.Concatenate("Features", "Open", "High", "Low", "Volume")
            .Append(mlContext.Transforms.CopyColumns("Label", "Close"))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label"));

        // Train the model
        var model = pipeline.Fit(data);

        // Predict the next day's closing price
        var predictionEngine = mlContext.Model.CreatePredictionEngine<HistoricalData, StockPrediction>(model);
        var lastDay = historicalData.Historical.Last();
        Console.WriteLine($"Last day's closing price for {symbol}: {lastDay.Close}");

        // Ensure lastDay is not null and has valid data
        if (lastDay == null)
        {
            return BadRequest("Invalid last day data.");
        }

        var prediction = predictionEngine.Predict(lastDay);

        Console.WriteLine($"Predicted closing price for {symbol}: {prediction.Close}");

        // Dummy recommendation logic
        var recommendation = prediction.Close > lastDay.Close ? "buy" : "sell";
        Console.WriteLine($"Recommendation: {recommendation}");

        return Ok(new { Symbol = symbol, Recommendation = recommendation });
    }

        private async Task<HistoricalDataResponse> LoadHistoricalData(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={apiKey}");
        Console.WriteLine(response);
        var settings = new JsonSerializerSettings();
        settings.Converters.Add(new HistoricalDataConverter());
        return JsonConvert.DeserializeObject<HistoricalDataResponse>(response, settings);
    }
}