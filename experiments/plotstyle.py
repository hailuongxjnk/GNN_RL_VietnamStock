"""Kiểu hình vẽ dùng chung cho luận văn (bảng màu phân loại cố định, lưới mờ, nét mảnh)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C1, C2, C3, C4, C5, C6, C7, C8 = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                   '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#104281']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.8, 'legend.frameon': False, 'figure.dpi': 120,
    'savefig.dpi': 220, 'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})
FIG = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.dirname(
    __import__('os').path.abspath(__file__))), 'thesis', 'figures')
