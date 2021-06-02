.PHONY: clean_search clean_info info_1 info_2 vi_1 vi_2 vi_3 vi_4

clean_search:
	rm -f cores/*.core outputs/search.*.out logs/search.*.log

clean_info:
	rm -f infos/info.*.txt outputs/info.*.out logs/info.*.log


info_1:
	<pomdps.txt ./batchify 40 1 | ./info.discovery

info_2:
	<pomdps.txt ./batchify 40 2 | ./info.discovery


vi_1:
	<pomdps.txt ./batchify 20 1 | ./vi.discovery

vi_2:
	<pomdps.txt ./batchify 20 2 | ./vi.discovery

vi_3:
	<pomdps.txt ./batchify 20 3 | ./vi.discovery

vi_4:
	<pomdps.txt ./batchify 20 4 | ./vi.discovery
