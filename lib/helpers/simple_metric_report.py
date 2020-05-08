def report(metrics):
    print("Metric report\n")
    for key, value in metrics.items():
        if isinstance(value, float):
            value = f"{value:.5f}"
        print(f"{key.capitalize()}: {value}")
