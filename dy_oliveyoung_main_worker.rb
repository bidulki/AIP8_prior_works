class DyOliveyoungMainWorker < HsDynamicWorker

    def parse_page(schedule_result, options)
      ## PARAM SET ##
      schedule_json = JSON.parse(schedule_result.schedule_json)
      prod_url = schedule_json['param1'].strip if schedule_json['param1'].present?
  
      get_browser.goto prod_url
      sleep 1
  
      review_button = get_browser.element(xpath: ".//li[@id='reviewInfo']")
      review_button.scroll.to :center
      review_button.click
      sleep 3
  
      sort_lis = get_browser.elements(xpath: ".//div[@class='align_sort']/ul/li")
  
      sort_lis.each do |sort_li|
        puts "#{sort_li.text}"
        sort_li.scroll.to :center
        sleep 1
        sort_li.click
        sleep 2
        while true
          lis = get_browser.elements(xpath: ".//div[@class='review_list_wrap']/ul/li")
          lis.last.scroll.to :center
          lis.each do |li|
            # user
            if li.element(xpath: ".//p[@class='info_user']/a").present?
              user = li.element(xpath: ".//p[@class='info_user']/a").text
            else
              raise HSRetryError.new('user 찾지 못함')
            end
            
            # 작성일
            if li.element(xpath: ".//span[@class='review_point']//span[@class='date']").present?
              write_date = li.element(xpath: ".//span[@class='review_point']//span[@class='date']").text
            else
              write_date = ""
            end
            
            # 평점
            if li.element(xpath: ".//span[@class='review_point']//span[@class='point']").present?
              review_point = li.element(xpath: ".//span[@class='review_point']//span[@class='point']").text.scan(/\d+/)[1].to_i
            else
              raise HSRetryError.new('평점 찾지 못함')
            end
  
            # 본문
            if li.element(xpath: ".//div[@class='txt_inner']").present?
              text = li.element(xpath: ".//div[@class='txt_inner']").text
            else
              next
            end
  
            # 카테고리
            category = {}
            ca_lis = nil
            if li.elements(xpath: ".//div[@class='poll_sample']/dl").present?
              ca_lis = li.elements(xpath: ".//div[@class='poll_sample']/dl")
            end
            if ca_lis != nil
              ca_lis.each do |ca_li|
                type = ca_li.element(xpath: ".//dt/span").text
                content = ca_li.element(xpath: ".//dd/span").text
                category[type] = content
              end
            end
  
            result = [review_point, text, category]
            uid = user + Digest::MD5.hexdigest(text) + write_date
  
            if schedule_result.exist_uid?(uid)
              puts "[SKIP] 이미 수집된 리뷰"
              next
            end
  
            begin
              make_result(schedule_result, [result], uid: uid)
            rescue ActiveRecord::RecordNotUnique => e
              ap e
            end
          end
  
          now_page = get_browser.element(xpath: ".//strong[@title='현재 페이지']")
  
          if now_page.next_sibling.present?
            # 다음페이지로 이동
            now_page.scroll.to :center
            now_page.next_sibling.click
            sleep 3
            puts '[알림]다음페이지로 넘어갑니다.'
          else
            puts '[알림]페이지네이션 다음항목이 존재하지 않습니다.수집을 종료합니다'
            break
          end
        end
      end
    end
  end
  