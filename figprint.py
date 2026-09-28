# -*- coding: utf-8 -*-
"""그림을 인쇄될 폭(mm)에 맞춰 그리고 저장하는 작은 도우미."""
import os

MM_PER_IN = 25.4


def figsize_mm(width_mm, aspect=0.62):
    """(가로인치, 세로인치). aspect는 세로/가로 비."""
    w = width_mm / MM_PER_IN
    return (w, w * aspect)


def save(fig, path, width_mm, dpi=300):
    """흰 바탕 PNG로 저장한다."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    return path
