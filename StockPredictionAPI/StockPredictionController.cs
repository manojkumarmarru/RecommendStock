
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.TimeSeries;
using Newtonsoft.Json;
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
    private const string alphaVantageApiKey = "V1DPYCL9VMBKG1SJ";
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
        // Prepare data for ML.NET
        var data = historicalData.Historical.Select(h => new StockData
        {
            Date = h.Date,
            Open = h.Open,
            High = h.High,
            Low = h.Low,
            Close = h.Close,
            Volume = h.Volume
        }).ToList();
        var dataView = mlContext.Data.LoadFromEnumerable(data);

        // Define forecasting pipeline
        var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
            outputColumnName: "ForecastedClose",
            inputColumnName: nameof(StockData.Close),
            windowSize: 30,
            seriesLength: data.Count,
            trainSize: data.Count,
            horizon: 1926,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: "LowerBoundClose",
            confidenceUpperBoundColumn: "UpperBoundClose");

        // Train the model
        var model = forecastingPipeline.Fit(dataView);
        // Create prediction engine
        var forecastingEngine = model.CreateTimeSeriesEngine<StockData, StockForecast>(mlContext);
        // Forecast the next 'years' years
        var forecast = forecastingEngine.Predict();
        // Prepare the result
        var forecastedData = new List<ForecastedData>();
        for (int i = 0; i < forecast.ForecastedClose.Length; i++)
        {
            forecastedData.Add(new ForecastedData
            {
                Date = data.Last().Date.AddDays(i + 1),
                ForecastedClose = forecast.ForecastedClose[i],
                LowerBoundClose = forecast.LowerBoundClose[i],
                UpperBoundClose = forecast.UpperBoundClose[i]
            });
        }
            forecastedData = forecastedData.OrderByDescending(f => f.Date).ToList();
        return Ok(new { Symbol = symbol, Forecast = forecastedData });
    }
    private async Task<HistoricalDataResponse> LoadHistoricalData(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={apiKey}");
        return JsonConvert.DeserializeObject<HistoricalDataResponse>(response);
    }
    private async Task<List<EMAData>> LoadEMAData(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://financialmodelingprep.com/api/v3/technical_indicator/1day/{symbol}?type=ema&period=1825&apikey={apiKey}");
        return JsonConvert.DeserializeObject<List<EMAData>>(response);
    }
    private async Task<List<SMAData>> LoadSMAData(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://financialmodelingprep.com/api/v3/technical_indicator/1day/{symbol}?type=sma&period=1825&apikey={apiKey}");
        return JsonConvert.DeserializeObject<List<SMAData>>(response);
    }
    private async Task<List<RSIData>> LoadRSIData(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://financialmodelingprep.com/api/v3/technical_indicator/1day/{symbol}?type=rsi&period=1825&apikey={apiKey}");
        return JsonConvert.DeserializeObject<List<RSIData>>(response);
    }
    private async Task<List<SentimentData>> LoadNewsSentiment(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={alphaVantageApiKey}");
        var sentimentResponse = JsonConvert.DeserializeObject<SentimentResponse>(response);
        return sentimentResponse?.Items ?? new List<SentimentData>();
    }
}
