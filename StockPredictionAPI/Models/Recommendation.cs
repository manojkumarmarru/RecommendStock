using System;
namespace StockPredictionAPI.Models
{
    public class Recommendation
    {
        public DateTime Date { get; set; }
        public float ForecastedClose { get; set; }
        public string Action { get; set; }
    }
}