import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(data, indicator_type, cmap, figsize=(10,10), annot=True, save=False):
  fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
  ax.set_title(f'{indicator_type.capitalize()} Indicators')
  sns.heatmap(data.corr(), annot = annot, fmt='.1g',vmin=-1, vmax=1, center= 0, linewidths=3, linecolor='black',square = True, cmap=cmap,cbar_kws={"shrink": .50})
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.5, top - 0.5)
  if save is True:
      plt.savefig(f'{indicator_type}_hmap.png', dpi=800)
  else:
      plt.plot()
