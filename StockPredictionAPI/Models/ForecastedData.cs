namespace StockPredictionAPI.Models
{
    public class ForecastedData
    {
        public DateTime Date { get; set; }
        public float ForecastedClose { get; set; }
        public float LowerBoundClose { get; set; }
        public float UpperBoundClose { get; set; }
    }
}