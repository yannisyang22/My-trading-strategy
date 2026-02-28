import matplotlib.pyplot as plt

def plot_equity_and_benchmark(bt, bh_equity, title="Equity Curve"):
    plt.figure()
    plt.plot(bt.index, bt["equity"], label="strategy")
    plt.plot(bt.index, bh_equity, label="buy&hold")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_drawdown(equity, title="Drawdown"):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    plt.figure()
    plt.plot(dd.index, dd)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()