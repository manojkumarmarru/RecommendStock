using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using StockPredictionAPI.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

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
    public async Task<IActionResult> GetStockPrediction(string symbol)
    {
        // Load historical data
        var historicalData = await LoadHistoricalData(symbol);
        if (historicalData == null || historicalData.Historical.Count == 0)
        {
            return BadRequest("No historical data available.");
        }

        // Load EMA data
        var emaData = await LoadEMAData(symbol);
        if (emaData == null || emaData.Count == 0)
        {
            return BadRequest("No EMA data available.");
        }

        // Merge EMA data with historical data
        foreach (var data in historicalData.Historical)
        {
            var ema = emaData.FirstOrDefault(e => e.Date == data.Date);
            if (ema != null)
            {
                data.EMA = ema.EMA;
            }
        }

        // Convert historical data to IDataView
        var dataView = mlContext.Data.LoadFromEnumerable(historicalData.Historical);

        // Define data preparation and model training pipeline
        var pipeline = mlContext.Transforms.Concatenate("Features", "Open", "High", "Low", "Volume", "EMA")
            .Append(mlContext.Transforms.CopyColumns("Label", "Close"))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label"));

        // Train the model
        var model = pipeline.Fit(dataView);

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

    private async Task<List<EMAData>> LoadEMAData(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://financialmodelingprep.com/api/v3/technical_indicator/1day/{symbol}?type=ema&&period=10&apikey={apiKey}");
        Console.WriteLine(response);
        return JsonConvert.DeserializeObject<List<EMAData>>(response);
    }
}