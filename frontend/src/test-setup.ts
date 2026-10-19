import '@testing-library/jest-dom/vitest'

Object.defineProperty(globalThis, 'ResizeObserver', {
	writable: true,
	value: class implements ResizeObserver {
		private callback: ResizeObserverCallback

		constructor(callback: ResizeObserverCallback) {
			this.callback = callback
		}

		observe(target: Element) {
			this.callback([{
				target,
				contentRect: { width: 800, height: 250 } as DOMRectReadOnly,
				borderBoxSize: [],
				contentBoxSize: [],
				devicePixelContentBoxSize: [],
			} as unknown as ResizeObserverEntry], this)
		}

		unobserve() {}
		disconnect() {}
	},
})