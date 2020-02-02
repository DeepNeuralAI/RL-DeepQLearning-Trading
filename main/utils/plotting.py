import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

def plot_heatmap(data, indicator_type, cmap, figsize=(10,10), annot=True, save=False):
	fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
	ax.set_title(f'{indicator_type.capitalize()} Indicators')
	sns.heatmap(data.corr(), annot = annot, fmt='.1g',vmin=-1, vmax=1, center= 0, \
    linewidths=3, linecolor='black',square = True, cmap=cmap,cbar_kws={"shrink": .50})
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5)
	if save is True: plt.savefig(f'{indicator_type}_hmap.png', dpi=800)
	return fig, ax


def plot_benchmarks(b_vals, p_vals, h_vals):
	fig, ax = plt.subplots(1, figsize=(12,6))
	ax.plot(b_vals, color = 'red', linewidth=1.5, label='Buy & Hold')
	ax.plot(p_vals, color = 'blue', linewidth=1.5, label='Perfect')
	ax.plot(h_vals, color = 'black', linewidth=1.5, label='Heuristic')
	ax.set_ylabel('Normalized Stock Values')
	ax.set_xlabel('Dates')
	ax.legend()
	fig.suptitle('Model Benchmark')
	return fig, ax

def visualize(df, history, title="trading session"):
    # add history to dataframe
    position = [history[0][0]] + [x[0] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df['position'] = position
    df['action'] = actions

    # specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(min(df['actual']), min(df['position'])) - 50, max(max(df['actual']), max(df['position'])) + 50), clamp=True)

    # plot a line chart for stock positions
    actual = alt.Chart(df).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale)
    ).interactive(
        bind_y=False
    )

    # plot the BUY and SELL actions as points
    points = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale),
        color='action'
    ).interactive(bind_y=False)

    # merge the two charts
    chart = alt.layer(actual, points, title=title).properties(height=300, width=1000)

    return chart
