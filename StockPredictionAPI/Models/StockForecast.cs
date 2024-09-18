namespace StockPredictionAPI.Models
{
    public class StockForecast
    {
        public float[] ForecastedClose { get; set; }
        public float[] LowerBoundClose { get; set; }
        public float[] UpperBoundClose { get; set; }
    }
}