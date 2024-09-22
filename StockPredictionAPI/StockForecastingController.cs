using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Newtonsoft.Json;
using NumSharp;
using NumSharp.Utilities;
using StockPredictionAPI.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

[Route("api/[controller]")]
[ApiController]
public class LSTMStockPredictionController : ControllerBase
{
    const string apiKey = "fdBhj6FJhbAKaYyTh5fU3pwUvPY5X32E";
    private const string vantageApiKey = "V1DPYCL9VMBKG1SJ";
    private readonly HttpClient _httpClient;
    private readonly MLContext mlContext;

    public LSTMStockPredictionController()
    {
        _httpClient = new HttpClient();
        mlContext = new MLContext();
    }

    [HttpGet("{symbol}")]
    public async Task<IActionResult> GetLSTMStockPrediction(string symbol, int years = 7)
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
            AdjClose = h.AdjClose,
            Volume = h.Volume
        }).ToList();

        // Sort the data by date
        var sortedData = data.OrderBy(d => d.Date).ToList();

        // Determine the split index
        int splitIndex = (int)(sortedData.Count * 0.9); // 90% train, 10% test

        // Split the data into training and test sets
        var trainDataList = sortedData.Take(splitIndex).ToList();
        var testDataList = sortedData.Skip(splitIndex).ToList();

        // Convert the split data into ML.NET data views
        var trainData = mlContext.Data.LoadFromEnumerable(trainDataList);
        var testData = mlContext.Data.LoadFromEnumerable(testDataList);

        // Process the data for LSTM
        var trainDataEnumerable = mlContext.Data.CreateEnumerable<StockData>(trainData, reuseRowObject: false).ToList();
        var testDataEnumerable = mlContext.Data.CreateEnumerable<StockData>(testData, reuseRowObject: false).ToList();

        var trainX = trainDataEnumerable.Select(d => new float[] { d.Open, d.High, d.Low, d.Volume }).ToArray();
        var testX = testDataEnumerable.Select(d => new float[] { d.Open, d.High, d.Low, d.Volume }).ToArray();

        var X_train = trainX.Select(x => new float[][] { x }).ToArray();
        var X_test = testX.Select(x => new float[][] { x }).ToArray();

        // Convert the data to the required format for Keras
        var y_train = trainDataEnumerable.Select(d => d.AdjClose).ToArray();
        var y_test = testDataEnumerable.Select(d => d.AdjClose).ToArray();

        // Save the processed data to files for use in Python
        SaveDataToFile(X_train, "X_train.npy");
        SaveDataToFile(X_test, "X_test.npy");
        SaveDataToFile(y_train, "y_train.npy");
        SaveDataToFile(y_test, "y_test.npy");

        Console.WriteLine($"Train Data starting date: {trainDataEnumerable.Min(d => d.Date)}");
        Console.WriteLine($"Train Data ending date: {trainDataEnumerable.Max(d => d.Date)}");
        Console.WriteLine($"Test Data starting date: {testDataEnumerable.Min(d => d.Date)}");
        Console.WriteLine($"Test Data ending date: {testDataEnumerable.Max(d => d.Date)}");

        // Extract the last date from the training data
        var lastDate = trainDataEnumerable.Max(d => d.Date);
        var startDate = lastDate.AddDays(1).ToString("yyyy-MM-dd");
        var forecastDays = (DateTime.Now - DateTime.Parse(startDate)).Days + 60;
        Console.WriteLine($"Start Date: {startDate}");
        Console.WriteLine($"Forecast Days: {forecastDays}");
        // Call the Python script to generate predictions
        var pythonScriptPath = "C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\build_lstm_model.py";
        var start = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "python",
            Arguments = $"{pythonScriptPath} {startDate} {forecastDays}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        using (var process = System.Diagnostics.Process.Start(start))
        {
            using (var reader = process.StandardOutput)
            {
                string pythonOutput = reader.ReadToEnd();
                Console.Write(pythonOutput);
            }
            using (var reader = process.StandardError)
            {
                string pythonError = reader.ReadToEnd();
                Console.Write(pythonError);
            }
        }

        // Load the combined dates and predictions from the Python script
        var datesFilePath = "C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\dates.txt";
        var predictionsFilePath = "C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python\\combined_predictions.npy";

        if (!System.IO.File.Exists(datesFilePath) || !System.IO.File.Exists(predictionsFilePath))
        {
            return StatusCode(500, "Dates or predictions file not found.");
        }

        var dates = System.IO.File.ReadAllLines(datesFilePath);

        var predictionsSlice = np.load(predictionsFilePath).GetData<float>();
        var predictions = predictionsSlice.ToArray();

        // Extract dates from historical data
        var historicalDates = sortedData.Select(d => d.Date).ToList();

        // Find the initial historicalIndex based on startDate
        int historicalIndex = historicalDates.FindIndex(date => date >= DateTime.Parse(startDate));
        if (historicalIndex == -1)
        {
            historicalIndex = historicalDates.Count; // If no date is found, start from the end of historical data
        }

        // Assign dates to predictions
        var forecastedDataWithLstm = new List<dynamic>();
        DateTime currentDate = DateTime.Today;

        foreach (var prediction in predictions)
        {
            // Use historical dates if available
            if (historicalIndex < historicalDates.Count && historicalDates[historicalIndex] >= DateTime.Parse(startDate))
            {
                forecastedDataWithLstm.Add(new
                {
                    ForecastedClose = prediction,
                    Date = historicalDates[historicalIndex].ToString("yyyy-MM-dd")
                });
                historicalIndex++;
            }
            else
            {
                // Skip weekends
                while (currentDate.DayOfWeek == DayOfWeek.Saturday || currentDate.DayOfWeek == DayOfWeek.Sunday)
                {
                    currentDate = currentDate.AddDays(1);
                }

                forecastedDataWithLstm.Add(new
                {
                    ForecastedClose = prediction,
                    Date = currentDate.ToString("yyyy-MM-dd")
                });
                currentDate = currentDate.AddDays(1);
            }
        }

        var forecastedAdjClose = predictions;
        // Print forecasted data for verification
        Console.WriteLine("Forecasted Data:", forecastedAdjClose);

        // Calculate accuracy metrics
        var actualValues = y_test;
        var mae = CalculateMAE(forecastedAdjClose, actualValues);
        var rmse = CalculateRMSE(forecastedAdjClose, actualValues);

        // Define metrics
        var metrics = new
        {
            MeanAbsoluteError = mae,
            RootMeanSquaredError = rmse
        };

        var trainDataRange = new {
            startDate = trainDataEnumerable.Min(d => d.Date).ToString("yyyy-MM-dd"),
            endDate = trainDataEnumerable.Max(d => d.Date).ToString("yyyy-MM-dd")
        };

        var testDataRange = new {
            startDate = testDataEnumerable.Min(d => d.Date).ToString("yyyy-MM-dd"),
            endDate = testDataEnumerable.Max(d => d.Date).ToString("yyyy-MM-dd")
        };

        // Prepare the result
        var resultData = new
        {
            Symbol = symbol,
            TrueAdjClose = data.Select(d => new { d.Date, d.AdjClose }),
            ForecastedAdjClose = forecastedDataWithLstm.Select(d => new { d.Date, d.ForecastedClose }),
            Metrics = metrics,
            trainDataRange,
            testDataRange
        };

        return Ok(resultData);
    }

    private void PrintDataFrameShapeAndCheckNulls(List<StockData> data)
    {
        int rowCount = data.Count;
        int columnCount = typeof(StockData).GetProperties().Length;

        bool hasNulls = data.Any(row => row.GetType().GetProperties().Any(prop => prop.GetValue(row) == null));

        Console.WriteLine($"Dataframe Shape: ({rowCount}, {columnCount})");
        Console.WriteLine($"Null Value Present: {hasNulls}");
    }

    private async Task<HistoricalDataResponse> LoadHistoricalData(string symbol)
    {
        var response = await _httpClient.GetStringAsync($"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={apiKey}");
        return JsonConvert.DeserializeObject<HistoricalDataResponse>(response);
    }

    private void SaveDataToFile(Array data, string fileName)
    {
        var filePath = Path.Combine("C:\\Users\\ManojKumarMarru\\source\\repos\\RecommendStock\\StockPredictionAPI\\python", fileName);
        var npData = np.array(data);
        np.save(filePath, npData);
    }

    private double CalculateMAE(float[] predictions, float[] actuals)
    {
        double sum = 0;
        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            sum += Math.Abs(predictions[i] - actuals[i]);
        }
        return sum / predictions.Length;
    }

    private double CalculateRMSE(float[] predictions, float[] actuals)
    {
        double sum = 0;
        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            sum += Math.Pow(predictions[i] - actuals[i], 2);
        }
        return Math.Sqrt(sum / predictions.Length);
    }
}